import random
import warnings
from copy import deepcopy
from functools import reduce
from itertools import chain
from operator import add
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
from typing_extensions import Literal

from lhotse import CutSet
from lhotse.cut import Cut
from lhotse.dataset.sampling.base import CutSampler, SamplingDiagnostics
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
        ...    max_frames=20000
        ... )

    Bucketing sampler with 20 buckets, sampling pairs of source-target cuts::

        >>> sampler = BucketingSampler(
        ...    cuts, target_cuts,
        ...    # BucketingSampler specific args
        ...    sampler_type=CutPairsSampler, num_buckets=20,
        ...    # Args passed into CutPairsSampler
        ...    max_source_frames=20000, max_target_frames=15000
        ... )
    """

    def __init__(
        self,
        *cuts: CutSet,
        sampler_type: Type = SimpleCutSampler,
        num_buckets: int = 10,
        bucket_method: Literal["equal_len", "equal_duration"] = "equal_len",
        drop_last: bool = False,
        proportional_sampling: bool = True,
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
        :param bucket_method: how should we shape the buckets. Available options are:
            "equal_len", where each bucket contains the same number of cuts,
            and "equal_duration", where each bucket has the same cumulative duration
            (but different number of cuts).
        :param drop_last: When ``True``, we will drop all incomplete batches.
            A batch is considered incomplete if it depleted a bucket before
            hitting the constraint such as max_duration, max_cuts, etc.
        :param proportional_sampling: When ``True``, we will introduce an approximate
            proportional sampling mechanism in the bucket selection.
            This mechanism reduces the chance that any of the buckets gets depleted early.
            Enabled by default.
        :param seed: random seed for bucket selection
        :param kwargs: Arguments used to create the underlying sampler for each bucket.
        """
        # Do not use the distributed capacities of the CutSampler in the top-level sampler.
        super().__init__(
            world_size=1,
            rank=0,
            seed=seed,
        )
        self.num_buckets = num_buckets
        self.drop_last = drop_last
        self.proportional_sampling = proportional_sampling
        self.sampler_type = sampler_type
        self.sampler_kwargs = kwargs
        self.cut_sets = cuts
        if self.cut_sets[0].is_lazy:
            warnings.warn(
                "Lazy CutSet detected in BucketingSampler: we will read it into memory anyway. "
                "Please use lhotse.dataset.DynamicBucketingSampler instead."
            )

        # Split data into buckets.
        self.bucket_method = bucket_method
        if self.bucket_method == "equal_len":
            self.buckets = create_buckets_equal_len(
                *self.cut_sets, num_buckets=num_buckets
            )
        elif self.bucket_method == "equal_duration":
            self.buckets = create_buckets_equal_duration(
                *self.cut_sets, num_buckets=num_buckets
            )
        else:
            raise ValueError(
                f"Unknown bucket_method: '{self.bucket_method}'. "
                f"Use one of: 'equal_len' or 'equal_duration'."
            )

        # Create a separate sampler for each bucket.
        self.bucket_samplers = [
            self.sampler_type(
                *bucket_cut_sets, drop_last=drop_last, **self.sampler_kwargs
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

    def allow_iter_to_reset_state(self):
        """
        Enables re-setting to the start of an epoch when iter() is called.
        This is only needed in one specific scenario: when we restored previous
        sampler state via ``sampler.load_state_dict()`` but want to discard
        the progress in the current epoch and start from the beginning.
        """
        super().allow_iter_to_reset_state()
        for s in self.bucket_samplers:
            s.allow_iter_to_reset_state()

    def state_dict(self) -> Dict[str, Any]:
        """
        Return the current state of the sampler in a state_dict.
        Together with ``load_state_dict()``, this can be used to restore the
        training loop's state to the one stored in the state_dict.
        """
        state_dict = super().state_dict()
        # We use deepcopies just in case somebody loads state dict during the same execution...
        state_dict.update(
            {
                "num_buckets": self.num_buckets,
                "drop_last": self.drop_last,
                "proportional_sampling": self.proportional_sampling,
                "bucket_method": self.bucket_method,
                "depleted": deepcopy(self.depleted),
                "bucket_samplers": [s.state_dict() for s in self.bucket_samplers],
                "sampler_kwargs": deepcopy(self.sampler_kwargs),
                "bucket_rng_state": self.bucket_rng.getstate(),
            }
        )
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Restore the state of the sampler that is described in a state_dict.
        This will result in the sampler yielding batches from where the previous training left it off.

        .. caution::
            The samplers are expected to be initialized with the same CutSets,
            but this is not explicitly checked anywhere.

        .. caution::
            The input ``state_dict`` is being mutated: we remove each consumed key, and expect
            it to be empty at the end of loading. If you don't want this behavior, pass a copy
            inside of this function (e.g., using ``import deepcopy``).

        .. note::
            For implementers of sub-classes of CutSampler: the flag ``self._just_restored_state`` has to be
            handled in ``__iter__`` to make it avoid resetting the just-restored state (only once).
        """
        num_buckets = state_dict.pop("num_buckets")
        assert self.num_buckets == num_buckets, (
            "Error in BucketingSampler.load_state_dict(): Inconsistent number of buckets: "
            f"current sampler has {self.num_buckets}, the state_dict has {num_buckets}."
        )
        self.drop_last = state_dict.pop("drop_last")
        self.proportional_sampling = state_dict.pop("proportional_sampling")
        self.bucket_method = state_dict.pop("bucket_method")
        self.sampler_kwargs = state_dict.pop("sampler_kwargs")
        self.depleted = state_dict.pop("depleted")
        self.bucket_rng.setstate(state_dict.pop("bucket_rng_state"))

        assert len(self.bucket_samplers) == len(state_dict["bucket_samplers"]), (
            "Error in BucketingSampler.load_state_dict(): Inconsistent number of samplers: "
            f"current sampler has {len(self.bucket_samplers)}, "
            f"the state_dict has {len(state_dict['bucket_samplers'])}."
        )
        for sampler, sampler_sd in zip(
            self.bucket_samplers, state_dict.pop("bucket_samplers")
        ):
            sampler.load_state_dict(sampler_sd)

        super().load_state_dict(state_dict)

    def __iter__(self) -> "BucketingSampler":
        # Restored state with load_state_dict()? Skip resetting.
        if self._just_restored_state:
            return self
        # Reset the state to the beginning of the epoch.
        self.bucket_rng.seed(self.seed + self.epoch)
        for b in self.bucket_samplers:
            iter(b)
        self.depleted = [False] * self.num_buckets
        return self

    def _select_bucket_with_idx(self) -> Tuple[int, CutSampler]:
        if not self.proportional_sampling or self.cut_sets[0].is_lazy:
            # Either proportional sampling was disabled, or the CutSet is lazy.
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
        self.allow_iter_to_reset_state()
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

    @property
    def diagnostics(self) -> SamplingDiagnostics:
        return reduce(add, (bucket.diagnostics for bucket in self.bucket_samplers))

    def get_report(self) -> str:
        """Returns a string describing the statistics of the sampling process so far."""
        return self.diagnostics.get_report()


def create_buckets_equal_len(
    *cuts: CutSet, num_buckets: int
) -> List[Tuple[CutSet, ...]]:
    """
    Creates buckets of cuts with similar durations.
    Each bucket has the same number of cuts, but different cumulative duration.

    :param cuts: One or more CutSets; the input CutSets are assumed to have the same cut IDs
        (i.e., the cuts correspond to each other and are meant to be sampled together as pairs,
        triples, etc.).
    :param num_buckets: The number of buckets.
    :return: A list of CutSet buckets (or tuples of CutSet buckets, depending on the input).
    """
    first_cut_set = cuts[0].sort_by_duration()
    buckets = [first_cut_set.split(num_buckets)] + [
        cs.sort_like(first_cut_set).split(num_buckets) for cs in cuts[1:]
    ]
    # zip(*buckets) does:
    # [(cs0_0, cs1_0, cs2_0), (cs0_1, cs1_1, cs2_1)] -> [(cs0_0, cs0_1), (cs1_0, cs1_1), (cs2_0, cs2_1)]
    buckets = list(zip(*buckets))
    return buckets


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
    total_duration = np.sum(c.duration for c in cuts)
    bucket_duration = total_duration / num_buckets
    iter_cuts = iter(cuts)
    buckets = []
    for bucket_idx in range(num_buckets):
        bucket = []
        current_duration = 0
        try:
            while current_duration < bucket_duration:
                bucket.append(next(iter_cuts))
                current_duration += bucket[-1].duration
            # Every odd bucket, take the cut that exceeded the bucket's duration
            # and put it in the front of the iterable, so that it goes to the
            # next bucket instead. It will ensure that the last bucket is not too
            # thin (otherwise all the previous buckets are a little too large).
            if bucket_idx % 2:
                last_cut = bucket.pop()
                iter_cuts = chain([last_cut], iter_cuts)
        except StopIteration:
            assert bucket_idx == num_buckets - 1
        buckets.append(CutSet.from_cuts(bucket))
    return buckets
