import random
import threading
import time
import warnings
from dataclasses import asdict, dataclass
from itertools import islice
from queue import Queue
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Generator,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import torch

from lhotse import CutSet, Seconds
from lhotse.cut import Cut
from lhotse.dataset.dataloading import resolve_seed
from lhotse.dataset.sampling.base import (
    CutSampler,
    EpochDiagnostics,
    SamplingConstraint,
    SamplingDiagnostics,
    TimeConstraint,
)
from lhotse.dataset.sampling.dynamic import DurationBatcher, Filter, check_constraint
from lhotse.utils import ifnone


class DynamicBucketingSampler(CutSampler):
    """
    A dynamic (streaming) variant of :class:`~lhotse.dataset.sampling.bucketing.BucketingSampler`,
    that doesn't require reading the whole cut set into memory.

    The basic idea is to sample N (e.g. ~10k) cuts and estimate the boundary durations for buckets.
    Then, we maintain a buffer of M cuts (stored separately in K buckets) and every time we sample a batch,
    we consume the input cut iterable for the same amount of cuts.
    The memory consumption is limited by M at all times.

    For scenarios such as ASR, VAD, Speaker ID, or TTS training, this class supports single CutSet
    iteration. Example::

        >>> cuts = CutSet(...)
        >>> sampler = DynamicBucketingSampler(cuts, max_duration=100)
        >>> for batch in sampler:
        ...     assert isinstance(batch, CutSet)

    For other scenarios that require pairs (or triplets, etc.) of utterances, this class supports
    zipping multiple CutSets together. Such scenarios could be voice conversion, speech translation,
    contrastive self-supervised training, etc. Example::

        >>> source_cuts = CutSet(...)
        >>> target_cuts = CutSet(...)
        >>> sampler = DynamicBucketingSampler(source_cuts, target_cuts, max_duration=100)
        >>> for batch in sampler:
        ...     assert isinstance(batch, tuple)
        ...     assert len(batch) == 2
        ...     assert isinstance(batch[0], CutSet)
        ...     assert isinstance(batch[1], CutSet)

    .. note:: for cut pairs, triplets, etc. the user is responsible for ensuring that the CutSets
        are all sorted so that when iterated over sequentially, the items are matched.
        We take care of preserving the right ordering internally, e.g., when shuffling.
        By default, we check that the cut IDs are matching, but that can be disabled.

    .. caution:: when using :meth:`DynamicBucketingSampler.filter` to filter some cuts with more than
        one CutSet to sample from, we sample one cut from every CutSet, and expect that all of the cuts
        satisfy the predicate -- otherwise, they are all discarded from being sampled.
    """

    def __init__(
        self,
        *cuts: Iterable,
        max_duration: Optional[Seconds] = None,
        max_cuts: Optional[int] = None,
        constraint: Optional[SamplingConstraint] = None,
        num_buckets: Optional[int] = 10,
        shuffle: bool = False,
        drop_last: bool = False,
        consistent_ids: bool = True,
        duration_bins: List[Seconds] = None,
        num_cuts_for_bins_estimate: int = 10000,
        buffer_size: int = 20000,
        quadratic_duration: Optional[Seconds] = None,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        seed: Union[int, Literal["randomized", "trng"]] = 0,
        sync_buckets: bool = True,
        concurrent: bool = False,
        strict=None,
        shuffle_buffer_size=None,
    ) -> None:
        """
        :param cuts: one or more CutSets (when more than one, will yield tuples of CutSets as mini-batches)
        :param max_duration: The maximum total recording duration from ``cuts``.
            Note: with multiple CutSets, ``max_duration`` constraint applies only to the first CutSet.
        :param max_cuts: The maximum total number of ``cuts`` per batch.
            When only ``max_duration`` is specified, this sampler yields static batch sizes.
        :param num_buckets: how many buckets to create. Ignored if duration_bins are provided.
        :param shuffle: When ``True``, the cuts will be shuffled dynamically with
            a reservoir-sampling-based algorithm.
            Convenient when mini-batch loop is inside an outer epoch-level loop, e.g.:
            `for epoch in range(10): for batch in dataset: ...` as every epoch will see a
            different cuts order.
        :param drop_last: When ``True``, we will drop all incomplete batches.
            A batch is considered incomplete if it depleted a bucket before
            hitting the constraint such as max_duration, max_cuts, etc.
        :param consistent_ids: Only affects processing of multiple CutSets.
            When ``True``, at each sampling step we check cuts from all CutSets have the same ID
            (i.e., the first cut from every CutSet should have the same ID, same for the second, third, etc.).
        :param duration_bins: A list of floats (seconds); when provided, we'll skip the initial
            estimation of bucket duration bins (useful to speed-up the launching of experiments).
        :param num_cuts_for_bins_estimate: We will draw this many cuts to estimate the duration bins
            for creating similar-duration buckets.
            Larger number means a better estimate to the data distribution, possibly at a longer init cost.
        :param buffer_size: How many cuts (or cut pairs, triplets) we hold at any time across all
            of the buckets.
            Increasing ``max_duration`` (batch_size) or ``num_buckets`` might require increasing this number.
            Larger number here will also improve shuffling capabilities.
            It will result in larger memory usage.
        :param quadratic_duration: When set, it adds an extra penalty that's quadratic in size w.r.t.
            a cuts duration. This helps get a more even GPU utilization across different input lengths
            when models have quadratic input complexity. Set between 15 and 40 for transformers.
        :param sync_buckets: When set, we'll try to make each DDP rank sample from as close
            duration buckets as possible to minimize the tail worker effect.
        :param concurrent: Enabling concurrency eliminates most of the waiting to pre-populate the
            bucketing buffers before the sampler starts yielding examples. For tarred/Lhotse Shar data
            this can speed up the start of the training. Note that enabling concurrency will cause the
            sampling results to be non-deterministic. This feature is experimental.
        :param world_size: Total number of distributed nodes. We will try to infer it by default.
        :param rank: Index of distributed node. We will try to infer it by default.
        :param seed: Random seed used to consistently shuffle the dataset across different processes.
        """
        super().__init__(
            drop_last=drop_last, world_size=world_size, rank=rank, seed=seed
        )
        if not all(cs.is_lazy for cs in cuts if isinstance(cs, CutSet)):
            warnings.warn(
                "You are using DynamicBucketingSampler with an eagerly read CutSet. "
                "You won't see any memory/speed benefits with that setup. "
                "Either use 'CutSet.from_jsonl_lazy' to read the CutSet lazily, or use a BucketingSampler instead."
            )
        self.cuts = cuts
        self.max_duration = max_duration
        self.max_cuts = max_cuts
        self.constraint = constraint
        self.shuffle = shuffle
        self.consistent_ids = consistent_ids
        self.num_cuts_for_bins_estimate = num_cuts_for_bins_estimate
        self.buffer_size = buffer_size
        self.quadratic_duration = quadratic_duration
        self.sync_buckets = sync_buckets
        self.concurrent = concurrent
        self.rng = None
        check_constraint(constraint, max_duration, max_cuts)

        if strict is not None:
            warnings.warn(
                "In Lhotse v1.4 all samplers act as if 'strict=True'. "
                "Sampler's argument 'strict' will be removed in a future Lhotse release.",
                category=DeprecationWarning,
            )

        if shuffle_buffer_size is not None:
            _emit_shuffle_buffer_size_warning()
            self.buffer_size += shuffle_buffer_size

        if duration_bins is not None:
            assert list(duration_bins) == sorted(
                duration_bins
            ), "Duration bins must be sorted ascendingly."
            self.duration_bins = duration_bins
            self.num_buckets = len(duration_bins) + 1
        else:
            if constraint is None:
                constraint = TimeConstraint(
                    max_duration=self.max_duration,
                    max_cuts=self.max_cuts,
                    quadratic_duration=self.quadratic_duration,
                )
            self.duration_bins = estimate_duration_buckets(
                islice(self.cuts[0], num_cuts_for_bins_estimate),
                num_buckets=num_buckets,
                constraint=constraint,
            )

    def state_dict(self) -> Dict[str, Any]:
        assert (
            self.constraint is None
        ), "state_dict() is not supported with samplers that use a custom constraint."
        sd = super().state_dict()
        sd.update(
            {
                "max_duration": self.max_duration,
                "max_cuts": self.max_cuts,
                "consistent_ids": self.consistent_ids,
                "buffer_size": self.buffer_size,
                "num_cuts_for_bins_estimate": self.num_cuts_for_bins_estimate,
                "quadratic_duration": self.quadratic_duration,
            }
        )
        return sd

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        self.max_duration = sd.pop("max_duration")
        self.max_cuts = sd.pop("max_cuts")
        self.consistent_ids = sd.pop("consistent_ids")
        self.num_cuts_for_bins_estimate = sd.pop("num_cuts_for_bins_estimate")
        self.buffer_size = sd.pop("buffer_size")
        if "shuffle_buffer_size" in sd:
            _emit_shuffle_buffer_size_warning()
            shuffle_buffer_size = sd.pop("shuffle_buffer_size")
            self.buffer_size += shuffle_buffer_size
        self.quadratic_duration = sd.pop("quadratic_duration", None)
        sd.pop("strict", None)  # backward compatibility
        super().load_state_dict(sd)
        self._fast_forward()

    def _fast_forward(self):
        current_epoch = self.diagnostics.current_epoch
        num_batches_to_iter = self.diagnostics.current_epoch_stats.total_batches

        # Set the right epoch
        self.set_epoch(current_epoch)
        # Reset diagnostics for this epoch as we're about to re-iterate
        self.diagnostics.stats_per_epoch[current_epoch] = EpochDiagnostics(
            epoch=current_epoch
        )

        self._just_restored_state = False
        iter(self)
        for _ in range(num_batches_to_iter):
            next(self)
        self._just_restored_state = True

    def __iter__(self) -> "DynamicBucketingSampler":
        if self._just_restored_state:
            return self
        seed = resolve_seed(self.seed)
        self.rng = random.Random(seed + self.epoch)
        if self.sync_buckets:
            # Bucket sync requested. To achieve that we will fix the RNG seed for a special bucket RNG
            # in a deterministic way. We also consider whether the sampler object lives in the training loop
            # process (map-style dataset or num_workers=0) or the dataloading subprocess (iterable-style dataset).
            # In the latter case, we want each worker to choose different buckets but still be in sync
            # with workers with the same IDs on other ranks.
            # Note: PyTorch dataloader always iterates workers sequentially, so they won't get out-of-order.
            bucket_rng_seed = 1234
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                bucket_rng_seed += worker_info.id
            bucket_rng = random.Random(bucket_rng_seed)
        else:
            bucket_rng = None
        # Why reset the current epoch?
        # Either we are iterating the epoch for the first time and it's a no-op,
        # or we are iterating the same epoch again, in which case setting more steps
        # than are actually available per epoch would have broken the checkpoint restoration.
        self.diagnostics.reset_current_epoch()
        # Initiate iteration
        cuts_iter = [iter(cs) for cs in self.cuts]
        # Apply filter predicate
        cuts_iter = Filter(
            iterator=zip(*cuts_iter),
            predicate=lambda tpl: all(self._filter_fn(c) for c in tpl),
            diagnostics=self.diagnostics,
        )
        # Convert Iterable[Cut] -> Iterable[CutSet]
        cuts_iter = DynamicBucketer(
            cuts_iter,
            duration_bins=self.duration_bins,
            world_size=self.world_size,
            max_duration=self.max_duration,
            max_cuts=self.max_cuts,
            constraint=self.constraint,
            drop_last=self.drop_last,
            buffer_size=self.buffer_size,
            quadratic_duration=self.quadratic_duration,
            shuffle=self.shuffle,
            rng=self.rng,
            bucket_rng=bucket_rng,
            concurrent=self.concurrent,
            diagnostics=self.diagnostics,
        )
        self.cuts_iter = iter(cuts_iter)
        return self

    def _next_batch(self) -> Union[CutSet, Tuple[CutSet]]:
        batch = next(self.cuts_iter)
        if self.consistent_ids and isinstance(batch, tuple):
            for cuts in zip(*batch):
                expected_id = cuts[0].id
                assert all(c.id == expected_id for c in cuts[1:]), (
                    f"The input CutSet are not sorted by cut ID in the same way. "
                    f"We sampled the following mismatched cut IDs: {', '.join(c.id for c in cuts)}. "
                    f"If this is expected, pass the argument 'consistent_ids=False' to DynamicBucketingSampler."
                )
        return batch

    @property
    def remaining_duration(self) -> Optional[float]:
        return None

    @property
    def remaining_cuts(self) -> Optional[int]:
        return None

    @property
    def num_cuts(self) -> Optional[int]:
        return None


@dataclass
class FixedBucketBatchSizeConstraint(SamplingConstraint):
    """
    Sampling constraint that accepts a pre-defined batch size for each bucket.
    It uses the example's sequence length to determine which bucket we're sampling for,
    and otherwise the batch size is locally static for each bucket.

    This constraint doesn't support samples longer than the upper bound of the last bucket;
    if such sample is provided, we will raise an exception.
    """

    max_seq_len_buckets: List[float]
    batch_sizes: List[int]
    current_bucket: Union[int, None] = None
    num_cuts: int = 0

    def __post_init__(self):
        assert sorted(self.max_seq_len_buckets) == list(self.max_seq_len_buckets)

    def is_active(self) -> bool:
        return True

    def add(self, example: Cut) -> None:
        """
        Increment the internal counter for the time constraint,
        selecting the right property from the input ``cut`` object.
        """
        seqlen = self.measure_length(example)
        bucket_idx = self.select_bucket(
            buckets=self.max_seq_len_buckets, example_len=seqlen
        )
        assert bucket_idx < len(self.max_seq_len_buckets), (
            f"Received example with sequence length {seqlen} that exceeds "
            f"the highest allowed length {self.max_seq_len_buckets[-1]}."
        )
        if self.current_bucket is None:
            self.current_bucket = bucket_idx
        else:
            assert self.current_bucket == bucket_idx, (
                f"User error: FixedBucketBatchSizeConstraint is supposed to be used only on one bucket. "
                f"The example we received has sequence length {seqlen} which is outside of the allowed bounds "
                f"for bucket index {bucket_idx} in buckets {self.max_seq_len_buckets}."
            )
        self.num_cuts += 1

    def exceeded(self) -> bool:
        """Is the constraint exceeded or not."""
        return self.num_cuts > self.batch_sizes[self.current_bucket]

    def close_to_exceeding(self) -> bool:
        """
        Check if the batch is close to satisfying the constraints.
        We define "closeness" as: if we added one more cut that has
        duration/num_frames/num_samples equal to the longest seen cut
        in the current batch, then the batch would have exceeded the constraints.
        """
        return self.num_cuts >= self.batch_sizes[self.current_bucket]

    def reset(self) -> None:
        """
        Reset the internal counter (to be used after a batch was created,
        to start collecting a new one).
        """
        self.current_bucket = None
        self.num_cuts = 0

    def measure_length(self, example: Cut) -> float:
        return example.duration

    def state_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.max_seq_len_buckets = state_dict.pop("max_seq_len_buckets")
        self.batch_sizes = state_dict.pop("batch_sizes")
        self.current_bucket = state_dict.pop("current_bucket")
        self.num_cuts = state_dict.pop("num_cuts")
        assert len(state_dict) == 0, (
            "Error in FixedBucketBatchSizeConstraint.load_state_dict(): Unexpected keys:\n- "
            + "\n- ".join(state_dict.keys())
        )

    def __add__(
        self, other: "FixedBucketBatchSizeConstraint"
    ) -> "FixedBucketBatchSizeConstraint":
        for key in ("max_seq_len_buckets", "batch_sizes", "current_bucket"):
            self_attr = getattr(self, key)
            other_attr = getattr(other, key)
            is_none = self_attr is None and other_attr is None
            assert is_none or self_attr == other_attr, (
                f"To add two TimeConstraint objects, they need to represent the same constraint "
                f"(got self.{key}={self_attr} != other.{key}={other_attr})."
            )
        return FixedBucketBatchSizeConstraint(
            max_seq_len_buckets=self.max_seq_len_buckets,
            batch_sizes=self.batch_sizes,
            current_bucket=self.current_bucket,
            num_cuts=self.num_cuts + other.num_cuts,
        )

    def __eq__(self, other: "TimeConstraint") -> bool:
        return (
            isinstance(other, FixedBucketBatchSizeConstraint)
            and self.max_seq_len_buckets == other.max_seq_len_buckets
            and self.batch_sizes == other.batch_sizes
            and self.current_bucket == other.current_bucket
        )


def estimate_duration_buckets(
    cuts: Iterable[Cut],
    num_buckets: int,
    constraint: Optional[SamplingConstraint] = None,
) -> List[float]:
    """
    Given an iterable of cuts and a desired number of buckets, select duration values
    that should start each bucket.

    The returned list, ``bins``, has ``num_buckets - 1`` elements.
    The first bucket should contain cuts with duration ``0 <= d < bins[0]``;
    the last bucket should contain cuts with duration ``bins[-1] <= d < float("inf")``,
    ``i``-th bucket should contain cuts with duration ``bins[i - 1] <= d < bins[i]``.

    :param cuts: an iterable of :class:`lhotse.cut.Cut`.
    :param num_buckets: desired number of buckets.
    :param constraint: object with ``.measure_length()`` method that's used to determine
        the size of each sample. If ``None``, we'll use ``TimeConstraint``.
    :return: a list of boundary duration values (floats).
    """
    assert num_buckets > 1

    if constraint is None:
        constraint = TimeConstraint()

    sizes = np.array([constraint.measure_length(c) for c in cuts])
    sizes.sort()
    assert num_buckets <= sizes.shape[0], (
        f"The number of buckets ({num_buckets}) must be smaller than "
        f"or equal to the number of cuts ({sizes.shape[0]})."
    )
    size_per_bucket = sizes.sum() / num_buckets

    bins = []
    tot = 0.0
    for size in sizes:
        if tot > size_per_bucket:
            bins.append(size)
            tot = 0.0
        tot += size

    return bins


class BucketSelectionState:
    """
    Helper class used in the context of bucket selection synchronization across DDP ranks.
    It's only necessary when using a map-style dataset (i.e., the sampler lives in the training loop process)
    and world_size is greater than 1. In these cases we have to use the same bucket idx ``world_size`` times
    to ensure each rank uses the same bucket. This is due to how CutSampler distributes mini-batches
    across ranks, ensuring the number of steps is always equal for each rank.
    """

    def __init__(
        self, bucket_rng: random.Random, num_buckets: int, world_size: int
    ) -> None:
        self._bucket_rng = bucket_rng
        self._num_buckets = num_buckets
        self._world_size = world_size
        self._usage_count = 0
        self._bucket_idx = None

    def select_bucket_idx(self) -> int:
        if self._bucket_idx is None or self._usage_count == self._world_size:
            self._bucket_idx = self._bucket_rng.randrange(self._num_buckets)
            self._usage_count = 0
        self._usage_count += 1
        return self._bucket_idx

    def save(self) -> Dict[str, Any]:
        return {
            "_bucket_rng": self._bucket_rng.getstate(),
            "_bucket_idx": self._bucket_idx,
            "_usage_count": self._usage_count,
        }

    def restore(self, ckpt: Dict[str, Any]) -> None:
        self._bucket_rng.setstate(ckpt["_bucket_rng"])
        self._bucket_idx = ckpt["_bucket_idx"]
        self._usage_count = ckpt["_usage_count"]


class DynamicBucketer:
    def __init__(
        self,
        cuts: Iterable[Union[Cut, Tuple[Cut]]],
        duration_bins: List[Seconds],
        world_size: int,
        max_duration: Optional[Seconds] = None,
        max_cuts: Optional[int] = None,
        constraint: Optional[SamplingConstraint] = None,
        drop_last: bool = False,
        buffer_size: int = 10000,
        quadratic_duration: Optional[Seconds] = None,
        shuffle: bool = False,
        rng: random.Random = None,
        bucket_rng: random.Random = None,
        concurrent: bool = False,
        diagnostics: Optional[SamplingDiagnostics] = None,
    ) -> None:
        self.cuts = cuts
        self.duration_bins = duration_bins
        self.world_size = world_size
        self.max_duration = max_duration
        self.max_cuts = max_cuts
        self.constraint = constraint
        self.drop_last = drop_last
        self.buffer_size = buffer_size
        self.quadratic_duration = quadratic_duration
        self.diagnostics = ifnone(diagnostics, SamplingDiagnostics())
        if rng is None:
            rng = random.Random()
        self.rng = rng
        self.bucket_rng = bucket_rng
        self.shuffle = shuffle
        self.concurrent = concurrent

        assert duration_bins == sorted(duration_bins), (
            f"Argument list for 'duration_bins' is expected to be in "
            f"sorted order (got: {duration_bins})."
        )
        check_constraint(constraint, max_duration, max_cuts)

        if self.constraint is None:
            self.constraint = TimeConstraint(
                max_duration=self.max_duration,
                max_cuts=self.max_cuts,
                quadratic_duration=self.quadratic_duration,
            )

        # A heuristic diagnostic first, for finding the right settings.
        if max_duration is not None:
            mean_duration = np.mean(duration_bins)
            expected_buffer_duration = buffer_size * mean_duration
            expected_bucket_duration = expected_buffer_duration / (
                len(duration_bins) + 1
            )
            if expected_bucket_duration < max_duration:
                warnings.warn(
                    f"Your 'buffer_size' setting of {buffer_size} might be too low to satisfy "
                    f"a 'max_duration' of {max_duration} (given our best guess)."
                )

        # Init: create empty buckets (note: `num_buckets = len(duration_bins) + 1`).
        self.buckets: List[Queue] = [Queue() for _ in range(len(duration_bins) + 1)]

        self._producer_thread = None

    def __iter__(self) -> Generator[CutSet, None, None]:
        # Init: sample `buffer_size` cuts and assign them to the right buckets.
        self.cuts_iter = iter(self.cuts)

        if self.concurrent:
            self._start_data_producer_thread()
            self._maybe_wait_for_producer()
        else:
            self._collect_cuts_in_buckets(self.buffer_size)

        state = BucketSelectionState(
            bucket_rng=self.bucket_rng,
            num_buckets=len(self.buckets),
            world_size=self.world_size,
        )

        # The iteration code starts here.
        # On each step we're sampling a new batch.
        try:
            while True:
                sampling_bucket = self._select_bucket(state)
                # Apply random shuffling if requested: we'll shuffle the items present within the bucket.
                maybe_shuffled = sampling_bucket
                indexes_used = []
                if self.shuffle:
                    maybe_shuffled = pick_at_random(
                        maybe_shuffled, rng=self.rng, out_indexes_used=indexes_used
                    )
                else:
                    with sampling_bucket.mutex:
                        maybe_shuffled = list(sampling_bucket.queue)
                # Sample one batch from that bucket and yield it to the caller.
                batcher = DurationBatcher(
                    maybe_shuffled,
                    constraint=self.constraint.copy(),
                    diagnostics=self.diagnostics,
                )
                batch = next(iter(batcher))
                if isinstance(batch, tuple):
                    batch_size = len(batch[0])
                else:
                    batch_size = len(batch)
                yield batch
                # Remove sampled cuts from the bucket.
                if indexes_used:
                    # Shuffling, sort indexes of yielded elements largest -> smallest and remove them
                    indexes_used.sort(reverse=True)
                    with sampling_bucket.mutex:
                        _q = sampling_bucket.queue
                        for idx in indexes_used:
                            del _q[idx]
                else:
                    # No shuffling, remove first N
                    for _ in range(batch_size):
                        sampling_bucket.get()
                # Fetch new cuts and add them to appropriate buckets.
                if self.concurrent:
                    self._maybe_wait_for_producer()
                else:
                    self._collect_cuts_in_buckets(batch_size)
        except StopIteration:
            pass

        # Cleanup.
        self.cuts_iter = None

    def _select_bucket(self, state: BucketSelectionState) -> Queue:
        if self.bucket_rng is None:
            # Bucket selection algo 1:
            # * there is just one RNG for choosing buckets and choosing samples randomly from the buckets
            # * check which buckets are ready, and then use the RNG to select one of them.
            # * no guarantees about bucket selection sync across GPUs.
            ready_buckets = [b for b in self.buckets if self._is_ready(b)]
            if not ready_buckets:
                # No bucket has enough data to yield for the last full batch.
                non_empty_buckets = [b for b in self.buckets if b]
                if self.drop_last or len(non_empty_buckets) == 0:
                    # Either the user requested only full batches, or we have nothing left.
                    raise StopIteration()
                else:
                    # Sample from partial batches that are left.
                    ready_buckets = non_empty_buckets
            # Choose a bucket to sample from.
            # We'll only select from the buckets that have a full batch available.
            return self.rng.choice(ready_buckets)
        else:
            # Bucket selection algo 2:
            # * bucket selection has its own independent RNG.
            # * when bucket selection RNG is initialized identically on all ranks/workers,
            #     then each rank will initially select the same bucket for batch sampling
            # * if one of the ranks selects a bucket that is not filled enough,
            #     it will scan the neighbouring buckets until it finds one that's ready
            # * if no bucket is ready, we end iteration

            def scan_buckets(predicate: Callable[[Queue], bool]) -> int:
                bucket_idx = state.select_bucket_idx()

                def valid_idx() -> bool:
                    return 0 <= bucket_idx < len(self.buckets)

                num_attempts = 0
                seen_min, seen_max = bucket_idx, bucket_idx
                while not (valid_idx() and predicate(self.buckets[bucket_idx])):
                    if seen_min < 0 and seen_max >= len(self.buckets):
                        raise BucketsDontHaveEnoughData()
                    num_attempts += 1
                    bucket_idx = (
                        bucket_idx + (1 if num_attempts % 2 == 0 else -1) * num_attempts
                    )
                    seen_min = min(seen_min, bucket_idx)
                    seen_max = max(seen_max, bucket_idx)

                return bucket_idx

            # This try/except is first trying to choose a bucket to sample a full mini-batch from,
            # and if that fails and drop_last=False, it tries again, this time accepting partial mini-batch.
            # Because we have no guarantee that samplers in different ranks will start exhausting the buckets
            # at the same time, it takes only a single occurrence of all buckets not being ready to permanently
            # run out-of-sync.
            # For this reason, we create a checkpoint of the bucket sampling state, and if we go into except
            # fallback, we restore this state first to ensure we use the bucket_rng exactly the same number
            # of times on each rank, no matter the circumstance.
            ckpt = state.save()
            try:
                # Typical case: at least one bucket has enough data to sample from.
                selected_bucket_idx = scan_buckets(self._is_ready)
            except BucketsDontHaveEnoughData:
                # We didn't hit the typical case either because we are finishing
                # the epoch, or because the buffers are too small.
                if self.drop_last:
                    # The user doesn't want partial mini-batches: early exit.
                    raise StopIteration()
                # The user wants to iterate the full dataset.
                # We'll try again, this time accepting buckets that have any amount of data available,
                # which may yield partial batches.
                try:
                    state.restore(ckpt)
                    selected_bucket_idx = scan_buckets(lambda b: b.qsize() > 0)
                except BucketsDontHaveEnoughData:
                    # We exhausted the full dataset.
                    raise StopIteration()

            return self.buckets[selected_bucket_idx]

    def _is_ready(self, bucket: Queue) -> bool:
        tot = self.constraint.copy()
        with bucket.mutex:
            contents = list(bucket.queue)
        for c in contents:
            tot.add(c[0] if isinstance(c, tuple) else c)
            if tot.close_to_exceeding():
                return True
        return False

    def _start_data_producer_thread(self):
        """Start concurrent filling of the bucket buffer in a background thread."""

        def producer():
            try:
                self._source_exhausted = False
                while not self._source_exhausted:
                    if sum(b.qsize() for b in self.buckets) == self.buffer_size:
                        time.sleep(0.1)
                        continue
                    cuts = next(self.cuts_iter)
                    bucket_idx = self.constraint.select_bucket(
                        buckets=self.duration_bins,
                        example=cuts[0] if isinstance(cuts, tuple) else cuts,
                    )
                    self.buckets[bucket_idx].put(cuts)
            except StopIteration:
                self._source_exhausted = True

        self._producer_thread = threading.Thread(target=producer)
        self._producer_thread.start()

    def _maybe_wait_for_producer(self):
        """Triggers wait for producer if the bucket buffers are less than 10% utilized."""
        while (
            sum(b.qsize() for b in self.buckets) < self.buffer_size / 10
            and not self._source_exhausted
        ):
            time.sleep(1.0)

    def _collect_cuts_in_buckets(self, n_cuts: int) -> None:
        """Fetches ``n_cuts`` from the input data iterable. Doesn't use concurrency."""
        try:
            for _ in range(n_cuts):
                cuts = next(self.cuts_iter)
                bucket_idx = self.constraint.select_bucket(
                    buckets=self.duration_bins,
                    example=cuts[0] if isinstance(cuts, tuple) else cuts,
                )
                self.buckets[bucket_idx].put(cuts)
        except StopIteration:
            pass

    def __del__(self):
        if (
            self.concurrent
            and self._producer_thread is not None
            and self._producer_thread.is_alive()
        ):
            self._source_exhausted = True
            self._producer_thread.join()


def pick_at_random(
    bucket: Queue,
    rng: random.Random,
    out_indexes_used: list,
) -> Generator[Union[Cut, Tuple[Cut, ...]], None, None]:
    """
    Generator which will yield items in a sequence in a random order.
    It will append the indexes of items yielded during iteration via ``out_used_indexes``.
    """
    with bucket.mutex:
        bucket = list(bucket.queue)
    indexes = list(range(len(bucket)))
    rng.shuffle(indexes)
    for idx in indexes:
        out_indexes_used.append(idx)
        yield bucket[idx]


class BucketsDontHaveEnoughData(Exception):
    pass


def _emit_shuffle_buffer_size_warning():
    warnings.warn(
        "Since Lhotse v1.20 'shuffle_buffer_size' is deprecated, because DynamicBucketingSampler "
        "does not require a separate shuffling buffer anymore. "
        "To improve both shuffling and sampling randomness, increase 'buffer_size' instead. "
        "To maintain backward compatibility, we will increase 'buffer_size' "
        "by 'shuffling_buffer_size' for you. "
        "This argument will be deprecated in a future Lhotse version.",
        category=DeprecationWarning,
    )
