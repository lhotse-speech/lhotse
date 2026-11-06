import random
from functools import reduce
from operator import add
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from lhotse import CutSet
from lhotse.cut import Cut
from lhotse.dataset.sampling.base import (
    CutSampler,
    SamplingDiagnostics,
    raise_checkpointing_unsupported,
)
from lhotse.dataset.sampling.simple import SimpleCutSampler


class BucketingSampler(CutSampler):
    """
    Sorts the cuts in a :class:`CutSet` by their duration and puts them into similar duration buckets.
    For each bucket, it instantiates a simpler sampler instance, e.g. :class:`SimpleCutSampler`.

    It behaves like an iterable that yields lists of strings (cut IDs).
    During iteration, it randomly selects one of the buckets to yield the batch from,
    until all the underlying samplers are depleted (which means it's the end of an epoch).

    Examples:

    Bucketing sampler with 20 buckets, sampling single cuts::

        >>> sampler = BucketingSampler(
        ...    cuts,
        ...    # BucketingSampler specific args
        ...    sampler_type=SimpleCutSampler, num_buckets=20,
        ...    # Args passed into SimpleCutSampler
        ...    max_duration=200
        ... )

    Bucketing sampler with 20 buckets, sampling pairs of source-target cuts::

        >>> sampler = BucketingSampler(
        ...    cuts, target_cuts,
        ...    # BucketingSampler specific args
        ...    sampler_type=CutPairsSampler, num_buckets=20,
        ...    # Args passed into CutPairsSampler
        ...    max_source_duration=200, max_target_duration=150
        ... )
    """

    def __init__(
        self,
        *cuts: CutSet,
        sampler_type: Type = SimpleCutSampler,
        num_buckets: int = 10,
        drop_last: bool = False,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        """
        BucketingSampler's constructor.

        :param cuts: one or more ``CutSet`` objects.
            The first one will be used to determine the buckets for all of them.
            Then, all of them will be used to instantiate the per-bucket samplers.
        :param sampler_type: a sampler type that will be created for each underlying bucket.
        :param num_buckets: how many buckets to create.
        :param drop_last: When ``True``, we will drop all incomplete batches.
            A batch is considered incomplete if it depleted a bucket before
            hitting the constraint such as max_duration, max_cuts, etc.
        :param seed: random seed for bucket selection
        :param kwargs: Arguments used to create the underlying sampler for each bucket.
        """
        # Do not use the distributed capacities of the CutSampler in the top-level sampler.
        super().__init__(
            drop_last=drop_last,
            world_size=1,
            rank=0,
            seed=seed,
        )
        self.num_buckets = num_buckets
        self.sampler_type = sampler_type
        self.sampler_kwargs = kwargs
        self.cut_sets = cuts
        if any(cs.is_lazy for cs in self.cut_sets):
            raise ValueError(
                "BucketingSampler does not support working with lazy CutSet (e.g., "
                "those opened with 'load_manifest_lazy', 'CutSet.from_jsonl_lazy', or "
                "'CutSet.from_webdataset'). "
                "Please use lhotse.dataset.DynamicBucketingSampler instead."
            )

        # Split data into buckets.
        self.buckets = create_buckets_equal_duration(
            *self.cut_sets, num_buckets=num_buckets
        )

        # Create a separate sampler for each bucket.
        self.bucket_samplers = [
            self.sampler_type(
                *bucket_cut_sets,
                drop_last=drop_last,
                **self.sampler_kwargs,
            )
            for bucket_cut_sets in self.buckets
        ]

        # Initialize mutable state.
        self.bucket_rng = random.Random(self.seed + self.epoch)
        self.depleted = [False] * num_buckets

    @property
    def remaining_duration(self) -> Optional[float]:
        """
        Remaining duration of data left in the sampler (may be inexact due to float arithmetic).
        Not available when the CutSet is read in lazy mode (returns None).

        .. note: For BucketingSampler, it's the sum of remaining duration in all buckets.
        """
        try:
            return sum(
                s.remaining_duration for _, s in self._nondepleted_samplers_with_idxs
            )
        except TypeError:
            return None

    @property
    def remaining_cuts(self) -> Optional[int]:
        """
        Remaining number of cuts in the sampler.
        Not available when the CutSet is read in lazy mode (returns None).

        .. note: For BucketingSampler, it's the sum of remaining cuts in all buckets.
        """
        try:
            return sum(
                s.remaining_cuts for _, s in self._nondepleted_samplers_with_idxs
            )
        except TypeError:
            return None

    @property
    def num_cuts(self) -> Optional[int]:
        """
        Total number of cuts in the sampler.
        Not available when the CutSet is read in lazy mode (returns None).

        .. note: For BucketingSampler, it's the sum of num cuts in all buckets.
        """
        try:
            return sum(s.num_cuts for s in self.bucket_samplers)
        except TypeError:
            return None

    def set_epoch(self, epoch: int) -> None:
        """
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        :param epoch: Epoch number.
        """
        for s in self.bucket_samplers:
            s.set_epoch(epoch)
        super().set_epoch(epoch)

    def filter(self, predicate: Callable[[Cut], bool]) -> None:
        """
        Add a constraint on individual cuts that has to be satisfied to consider them.

        Can be useful when handling large, lazy manifests where it is not feasible to
        pre-filter them before instantiating the sampler.

        Example:
            >>> cuts = CutSet(...)
            ... sampler = SimpleCutSampler(cuts, max_duration=100.0)
            ... # Retain only the cuts that have at least 1s and at most 20s duration.
            ... sampler.filter(lambda cut: 1.0 <= cut.duration <= 20.0)
        """
        for sampler in self.bucket_samplers:
            sampler.filter(predicate)

    def state_dict(self) -> Dict[str, Any]:
        raise_checkpointing_unsupported(self)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        raise_checkpointing_unsupported(self)

    def __iter__(self) -> "BucketingSampler":
        # Why reset the current epoch?
        # Either we are iterating the epoch for the first time and it's a no-op,
        # or we are iterating the same epoch again, in which case setting more steps
        # than are actually available per epoch would have broken the checkpoint restoration.
        self.diagnostics.reset_current_epoch()
        # Reset the state to the beginning of the epoch.
        self.bucket_rng.seed(self.seed + self.epoch)
        for b in self.bucket_samplers:
            iter(b)
        self.depleted = [False] * self.num_buckets
        return self

    def _select_bucket_with_idx(self) -> Tuple[int, CutSampler]:
        if self.cut_sets[0].is_lazy:
            # With lazy CutSets, we simply choose a random bucket,
            # because we can't know how much data is left in the buckets.
            return self.bucket_rng.choice(self._nondepleted_samplers_with_idxs)
        idx_sampler_pairs = self._nondepleted_samplers_with_idxs
        if len(idx_sampler_pairs) == 1:
            # Only a single bucket left -- choose it.
            return idx_sampler_pairs[0]
        # If we got there, it means there are at least 2 buckets we can sample from.
        # We are going to use approximate proportional sampling:
        # for that, we randomly select two buckets, and then assign a higher probability
        # to the bucket that has more cumulative data duration left to sample.
        # This helps ensure that none of the buckets is depleted much earlier than
        # the others.
        idx1, sampler1 = self.bucket_rng.choice(idx_sampler_pairs)
        idx2, sampler2 = self.bucket_rng.choice(idx_sampler_pairs)
        # Note: prob1 is the probability of selecting sampler1
        try:
            prob1 = sampler1.remaining_duration / (
                sampler1.remaining_duration + sampler2.remaining_duration
            )
        except ZeroDivisionError:
            # This will happen when we have already depleted the samplers,
            # but the BucketingSampler doesn't know it yet. We only truly
            # know that a sampler is depleted when we try to get a batch
            # and it raises a StopIteration, which is done after this stage.
            # We can't depend on remaining_duration for lazy CutSets.
            # If both samplers are zero duration, just return the first one.
            return idx1, sampler1
        if self.bucket_rng.random() > prob1:
            return idx2, sampler2
        else:
            return idx1, sampler1

    def _next_batch(self):
        while not self.is_depleted:
            idx, sampler = self._select_bucket_with_idx()
            try:
                return next(sampler)
            except StopIteration:
                self.depleted[idx] = True
        raise StopIteration()

    @property
    def is_depleted(self) -> bool:
        return all(self.depleted)

    @property
    def _nondepleted_samplers_with_idxs(self):
        return [
            (idx, bs)
            for idx, (bs, depleted) in enumerate(
                zip(self.bucket_samplers, self.depleted)
            )
            if not depleted
        ]

    def _log_diagnostics(self, batch: Union[CutSet, Tuple[CutSet, ...]]) -> None:
        return  # do nothing

    @property
    def diagnostics(self) -> SamplingDiagnostics:
        return reduce(add, (bucket.diagnostics for bucket in self.bucket_samplers))

    def get_report(self) -> str:
        """Returns a string describing the statistics of the sampling process so far."""
        return self.diagnostics.get_report()


def create_buckets_equal_duration(
    *cuts: CutSet, num_buckets: int
) -> List[Tuple[CutSet, ...]]:
    """
    Creates buckets of cuts with similar durations.
    Each bucket has the same cumulative duration, but a different number of cuts.

    :param cuts: One or more CutSets; the input CutSets are assumed to have the same cut IDs
        (i.e., the cuts correspond to each other and are meant to be sampled together as pairs,
        triples, etc.).
    :param num_buckets: The number of buckets.
    :return: A list of CutSet buckets (or tuples of CutSet buckets, depending on the input).
    """
    first_cut_set = cuts[0].sort_by_duration(ascending=True)
    buckets_per_cutset = [
        _create_buckets_equal_duration_single(first_cut_set, num_buckets=num_buckets)
    ]
    for cut_set in cuts[1:]:
        buckets_per_cutset.append(
            # .subset() will cause the output CutSet to have the same order of cuts as `bucket`
            cut_set.subset(cut_ids=bucket.ids)
            for bucket in buckets_per_cutset[0]
        )
    # zip(*buckets) does:
    # [(cs0_0, cs1_0, cs2_0), (cs0_1, cs1_1, cs2_1)] -> [(cs0_0, cs0_1), (cs1_0, cs1_1), (cs2_0, cs2_1)]
    return list(zip(*buckets_per_cutset))


def _create_buckets_equal_duration_single(
    cuts: CutSet, num_buckets: int
) -> List[CutSet]:
    """
    Helper method to partition a single CutSet into buckets that have the same
    cumulative duration.

    See also: :meth:`.create_buckets_from_duration_percentiles`.
    """
    total_duration = np.sum([c.duration for c in cuts])
    bucket_duration = total_duration / num_buckets
    # Define the order for adding cuts. We start at the beginning, then go to
    # the end, and work our way to the middle. Once in the middle we distribute
    # excess cuts among the two buckets close to the median duration. This
    # handles the problem of where to place cuts that caused previous buckets
    # to "over-flow" without sticking all of them in the last bucket, which
    # causes one large bucket at the end and also places many small duration
    # cuts with longer ones.
    order = list(range(0, len(cuts), 2)) + list(
        range(len(cuts) - (1 + len(cuts) % 2), 0, -2)
    )
    order2idx = {o_idx: i for i, o_idx in enumerate(order)}
    durations = [c.duration for c in cuts]

    # We need a list of the cut durations in the same order (0, N-1, 1, N-2, ...)
    ordered_cut_durations = sorted(zip(order, durations), key=lambda x: x[0])
    last_order, first_bucket = 0, 0
    last_bucket = num_buckets - 1
    buckets_dict = {i: 0 for i in range(num_buckets)}
    buckets_cut_dict = {i: [] for i in range(num_buckets)}
    middle_bucket = None
    idx_to_bucket_id = {}
    for i, (order_idx, duration) in enumerate(ordered_cut_durations, 1):
        # Check if we are at the middle bucket. first_bucket is the left bucket
        # we are processing. last_bucket is the right bucket. When they are the
        # same we are filling the bucket with cuts near the median duration.
        if middle_bucket is None and first_bucket == last_bucket:
            middle_bucket = first_bucket

        # i % 2 = 1 ==> process the left_bucket (first_bucket)
        if i % 2:
            if buckets_dict[first_bucket] + duration > bucket_duration:
                if middle_bucket is not None and first_bucket == middle_bucket:
                    first_bucket = max(0, min(middle_bucket - 1, num_buckets - 1))
                else:
                    first_bucket = min(first_bucket + 1, num_buckets - 1)
            buckets_dict[first_bucket] += duration
            idx_to_bucket_id[order2idx[order_idx]] = first_bucket
        # i % 2 = 0 ==> process the right bucket (last_bucket)
        else:
            if buckets_dict[last_bucket] + duration > bucket_duration:
                if middle_bucket is not None and last_bucket == middle_bucket:
                    last_bucket = max(middle_bucket + 1, 0)
                else:
                    last_bucket = max(last_bucket - 1, 0)
            buckets_dict[last_bucket] += duration
            idx_to_bucket_id[order2idx[order_idx]] = last_bucket

    # Now that buckets have been assigned, create the new cutset.
    for cut_idx, cut in enumerate(cuts):
        buckets_cut_dict[idx_to_bucket_id[cut_idx]].append(cut)
    buckets = [CutSet.from_cuts(buckets_cut_dict[i]) for i in range(num_buckets)]
    return buckets
