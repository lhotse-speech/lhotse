import random
import warnings
from bisect import bisect_right
from collections import deque
from itertools import islice
from typing import (
    Deque,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np

from lhotse import CutSet, Seconds
from lhotse.cut import Cut
from lhotse.dataset.sampling.base import CutSampler, SamplingDiagnostics, TimeConstraint
from lhotse.dataset.sampling.dynamic import DurationBatcher, Filter
from lhotse.utils import ifnone, streaming_shuffle


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
        *cuts: CutSet,
        max_duration: float,
        num_buckets: int = 10,
        shuffle: bool = False,
        drop_last: bool = False,
        consistent_ids: bool = True,
        num_cuts_for_bins_estimate: int = 10000,
        buffer_size: int = 10000,
        shuffle_buffer_size: int = 20000,
        strict: bool = False,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        """
        :param cuts: one or more CutSets (when more than one, will yield tuples of CutSets as mini-batches)
        :param max_duration: The maximum total recording duration from ``cuts``.
            Note: with multiple CutSets, ``max_duration`` constraint applies only to the first CutSet.
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
        :param num_cuts_for_bins_estimate: We will draw this many cuts to estimate the duration bins
            for creating similar-duration buckets.
            Larger number means a better estimate to the data distribution, possibly at a longer init cost.
        :param buffer_size: How many cuts (or cut pairs, triplets) we hold at any time across all
            of the buckets.
            Increasing ``max_duration`` (batch_size) or ``num_buckets`` might require increasing this number.
            It will result in larger memory usage.
        :param shuffle_buffer_size: How many cuts (or cut pairs, triplets) are being held in memory
            a buffer used for streaming shuffling. Larger number means better randomness at the cost
            of higher memory usage.
        :param strict: When ``True``, for the purposes of determining dynamic batch size,
            we take the longest cut sampled so far and multiply its duration/num_frames/num_samples
            by the number of cuts currently in mini-batch to check if it exceeded max_duration/etc.
            This can help make the GPU memory usage more predictable when there is a large variance
            in cuts duration.
        :param world_size: Total number of distributed nodes. We will try to infer it by default.
        :param rank: Index of distributed node. We will try to infer it by default.
        :param seed: Random seed used to consistently shuffle the dataset across different processes.
        """
        super().__init__(world_size=world_size, rank=rank, seed=seed)
        if not all(cs.is_lazy for cs in cuts if isinstance(cs, CutSet)):
            warnings.warn(
                "You are using DynamicBucketingSampler with an eagerly read CutSet. "
                "You won't see any memory/speed benefits with that setup. "
                "Either use 'CutSet.from_jsonl_lazy' to read the CutSet lazily, or use a BucketingSampler instead."
            )
        self.cuts = cuts
        self.max_duration = max_duration
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.consistent_ids = consistent_ids
        self.num_cuts_for_bins_estimate = num_cuts_for_bins_estimate
        self.buffer_size = buffer_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.strict = strict
        self.rng = None

        if self.shuffle:
            cuts_for_bins_estimate = streaming_shuffle(
                iter(self.cuts[0]),
                rng=random.Random(self.seed),
                bufsize=self.shuffle_buffer_size,
            )
        else:
            cuts_for_bins_estimate = self.cuts[0]
        self.duration_bins = estimate_duration_buckets(
            islice(cuts_for_bins_estimate, num_cuts_for_bins_estimate),
            num_buckets=num_buckets,
        )

    def __iter__(self) -> "DynamicBucketingSampler":
        self.rng = random.Random(self.seed + self.epoch)
        # Initiate iteration
        self.cuts_iter = [iter(cs) for cs in self.cuts]
        # Optionally shuffle
        if self.shuffle:
            self.cuts_iter = [
                # Important -- every shuffler has a copy of RNG seeded in the same way,
                # so that they are reproducible.
                streaming_shuffle(
                    cs,
                    rng=random.Random(self.seed + self.epoch),
                    bufsize=self.shuffle_buffer_size,
                )
                for cs in self.cuts_iter
            ]
        # Apply filter predicate
        self.cuts_iter = Filter(
            iterator=zip(*self.cuts_iter),
            predicate=lambda tpl: all(self._filter_fn(c) for c in tpl),
            diagnostics=self.diagnostics,
        )
        # Convert Iterable[Cut] -> Iterable[CutSet]
        self.cuts_iter = DynamicBucketer(
            self.cuts_iter,
            duration_bins=self.duration_bins,
            max_duration=self.max_duration,
            drop_last=self.drop_last,
            buffer_size=self.buffer_size,
            strict=self.strict,
            rng=self.rng,
        )
        self.cuts_iter.diagnostics = self.diagnostics
        self.cuts_iter = iter(self.cuts_iter)
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


def estimate_duration_buckets(cuts: Iterable[Cut], num_buckets: int) -> List[Seconds]:
    """
    Given an iterable of cuts and a desired number of buckets, select duration values
    that should start each bucket.

    The returned list, ``bins``, has ``num_buckets - 1`` elements.
    The first bucket should contain cuts with duration ``0 <= d < bins[0]``;
    the last bucket should contain cuts with duration ``bins[-1] <= d < float("inf")``,
    ``i``-th bucket should contain cuts with duration ``bins[i - 1] <= d < bins[i]``.

    :param cuts: an iterable of :class:`lhotse.cut.Cut`.
    :param num_buckets: desired number of buckets.
    :return: a list of boundary duration values (floats).
    """
    assert num_buckets > 1

    durs = np.array([c.duration for c in cuts])
    durs.sort()
    assert num_buckets < durs.shape[0], (
        f"The number of buckets ({num_buckets}) must be smaller "
        f"than the number of cuts ({durs.shape[0]})."
    )
    bucket_duration = durs.sum() / num_buckets

    bins = []
    tot = 0.0
    for dur in durs:
        if tot > bucket_duration:
            bins.append(dur)
            tot = 0.0
        tot += dur

    return bins


class DynamicBucketer:
    def __init__(
        self,
        cuts: Iterable[Union[Cut, Tuple[Cut]]],
        duration_bins: List[Seconds],
        max_duration: float,
        drop_last: bool = False,
        buffer_size: int = 10000,
        strict: bool = False,
        rng: random.Random = None,
        diagnostics: Optional[SamplingDiagnostics] = None,
    ) -> None:
        self.cuts = cuts
        self.duration_bins = duration_bins
        self.max_duration = max_duration
        self.drop_last = drop_last
        self.buffer_size = buffer_size
        self.strict = strict
        self.diagnostics = ifnone(diagnostics, SamplingDiagnostics())
        if rng is None:
            rng = random.Random()
        self.rng = rng

        assert duration_bins == sorted(duration_bins), (
            f"Argument list for 'duration_bins' is expected to be in "
            f"sorted order (got: {duration_bins})."
        )

        # A heuristic diagnostic first, for finding the right settings.
        mean_duration = np.mean(duration_bins)
        expected_buffer_duration = buffer_size * mean_duration
        expected_bucket_duration = expected_buffer_duration / (len(duration_bins) + 1)
        if expected_bucket_duration < max_duration:
            warnings.warn(
                f"Your 'buffer_size' setting of {buffer_size} might be too low to satisfy "
                f"a 'max_duration' of {max_duration} (given our best guess)."
            )

        # Init: create empty buckets (note: `num_buckets = len(duration_bins) + 1`).
        self.buckets: List[Deque[Union[Cut, Tuple[Cut]]]] = [
            deque() for _ in range(len(duration_bins) + 1)
        ]

    def __iter__(self) -> Generator[CutSet, None, None]:
        # Init: sample `buffer_size` cuts and assign them to the right buckets.
        self.cuts_iter = iter(self.cuts)
        self._collect_cuts_in_buckets(self.buffer_size)

        # Init: determine which buckets are "ready"
        def is_ready(bucket: Deque[Cut]):
            tot = TimeConstraint(max_duration=self.max_duration, strict=self.strict)
            for c in bucket:
                tot.add(c[0] if isinstance(c, tuple) else c)
                if tot.close_to_exceeding():
                    return True
            return False

        assert any(is_ready(bucket) for bucket in self.buckets)

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
                # Sample one batch from that bucket and yield it to the caller.
                batcher = DurationBatcher(
                    sampling_bucket,
                    max_duration=self.max_duration,
                    diagnostics=self.diagnostics,
                )
                batch = next(iter(batcher))
                if isinstance(batch, tuple):
                    batch_size = len(batch[0])
                else:
                    batch_size = len(batch)
                yield batch
                # Remove sampled cuts from the bucket.
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
                duration = (
                    cuts[0].duration if isinstance(cuts, tuple) else cuts.duration
                )
                bucket_idx = bisect_right(self.duration_bins, duration)
                self.buckets[bucket_idx].append(cuts)
        except StopIteration:
            pass
