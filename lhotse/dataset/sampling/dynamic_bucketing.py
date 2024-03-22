import random
import warnings
from bisect import bisect_right
from collections import deque
from itertools import islice
from typing import (
    Any,
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
        strict=None,
        shuffle_buffer_size=None,
    ) -> None:
        """
        :param cuts: one or more CutSets (when more than one, will yield tuples of CutSets as mini-batches)
        :param max_duration: The maximum total recording duration from ``cuts``.
            Note: with multiple CutSets, ``max_duration`` constraint applies only to the first CutSet.
        :param max_cuts: The maximum total number of ``cuts`` per batch.
            When only ``max_duration`` is specified, this sampler yields static batch sizes.
        :param num_buckets: how many buckets to create.
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
            if num_buckets is not None:
                assert len(duration_bins) == num_buckets - 1, (
                    f"num_buckets=={num_buckets} but len(duration_bins)=={len(duration_bins)} "
                    f"(bins are the boundaries, it should be one less than the number of buckets)."
                )
            assert list(duration_bins) == sorted(
                duration_bins
            ), "Duration bins must be sorted ascendingly."
            self.duration_bins = duration_bins
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
            max_duration=self.max_duration,
            max_cuts=self.max_cuts,
            constraint=self.constraint,
            drop_last=self.drop_last,
            buffer_size=self.buffer_size,
            quadratic_duration=self.quadratic_duration,
            shuffle=self.shuffle,
            rng=self.rng,
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


class DynamicBucketer:
    def __init__(
        self,
        cuts: Iterable[Union[Cut, Tuple[Cut]]],
        duration_bins: List[Seconds],
        max_duration: Optional[Seconds] = None,
        max_cuts: Optional[int] = None,
        constraint: Optional[SamplingConstraint] = None,
        drop_last: bool = False,
        buffer_size: int = 10000,
        quadratic_duration: Optional[Seconds] = None,
        shuffle: bool = False,
        rng: random.Random = None,
        diagnostics: Optional[SamplingDiagnostics] = None,
    ) -> None:
        self.cuts = cuts
        self.duration_bins = duration_bins
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
        self.shuffle = shuffle

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
        self.buckets: List[Deque[Union[Cut, Tuple[Cut, ...]]]] = [
            deque() for _ in range(len(duration_bins) + 1)
        ]

    def __iter__(self) -> Generator[CutSet, None, None]:
        # Init: sample `buffer_size` cuts and assign them to the right buckets.
        self.cuts_iter = iter(self.cuts)
        self._collect_cuts_in_buckets(self.buffer_size)

        # Init: determine which buckets are "ready"
        def is_ready(bucket: Deque[Cut]):
            tot = self.constraint.copy()
            for c in bucket:
                tot.add(c[0] if isinstance(c, tuple) else c)
                if tot.close_to_exceeding():
                    return True
            return False

        # The iteration code starts here.
        # On each step we're sampling a new batch.
        try:
            while True:
                ready_buckets = [b for b in self.buckets if is_ready(b)]
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
                sampling_bucket = self.rng.choice(ready_buckets)
                # Apply random shuffling if requested: we'll shuffle the items present within the bucket.
                maybe_shuffled = sampling_bucket
                indexes_used = []
                if self.shuffle:
                    maybe_shuffled = pick_at_random(
                        maybe_shuffled, rng=self.rng, out_indexes_used=indexes_used
                    )
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
                    for idx in indexes_used:
                        del sampling_bucket[idx]
                else:
                    # No shuffling, remove first N
                    for _ in range(batch_size):
                        sampling_bucket.popleft()
                # Fetch new cuts and add them to appropriate buckets.
                self._collect_cuts_in_buckets(batch_size)
        except StopIteration:
            pass

        # Cleanup.
        self.cuts_iter = None

    def _collect_cuts_in_buckets(self, n_cuts: int):
        try:
            for _ in range(n_cuts):
                cuts = next(self.cuts_iter)
                duration = self.constraint.measure_length(
                    cuts[0] if isinstance(cuts, tuple) else cuts
                )
                bucket_idx = bisect_right(self.duration_bins, duration)
                self.buckets[bucket_idx].append(cuts)
        except StopIteration:
            pass


def pick_at_random(
    bucket: Sequence[Union[Cut, Tuple[Cut, ...]]],
    rng: random.Random,
    out_indexes_used: list,
) -> Generator[Union[Cut, Tuple[Cut, ...]], None, None]:
    """
    Generator which will yield items in a sequence in a random order.
    It will append the indexes of items yielded during iteration via ``out_used_indexes``.
    """
    indexes = list(range(len(bucket)))
    rng.shuffle(indexes)
    for idx in indexes:
        out_indexes_used.append(idx)
        yield bucket[idx]


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
