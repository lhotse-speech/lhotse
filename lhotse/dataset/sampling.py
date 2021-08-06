import random
import warnings
from collections import deque
from dataclasses import dataclass, field
from functools import reduce
from math import isclose
from operator import add
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Type, Union

import torch.distributed as dist
from torch.utils.data import Sampler

from lhotse import CutSet
from lhotse.cut import Cut
from lhotse.utils import Seconds, exactly_one_not_null, is_none_or_gt


class DataSource:
    def __init__(self, items: CutSet):
        self._orig_items = items
        self._shuffled_items = self._orig_items
        self._iter = None
        self._reusable = deque()

    def shuffle(self, seed):
        r = random.Random(seed)
        self._shuffled_items = self._orig_items.shuffle(rng=r)
        self._iter = None
        self._reusable.clear()

    def sort_like(self, other: 'DataSource'):
        self._shuffled_items = self._orig_items.sort_like(other._shuffled_items)
        self._iter = None
        self._reusable.clear()

    def __iter__(self):
        self._iter = iter(self._shuffled_items)
        self._reusable.clear()
        return self

    def __next__(self):
        if self._reusable:
            return self._reusable.popleft()
        return next(self._iter)

    def take_back(self, cut: Cut) -> None:
        self._reusable.append(cut)

    def __len__(self):
        return len(self._shuffled_items)


class CutSampler(Sampler):
    """
    ``CutSampler`` is responsible for collecting batches of cuts, given specified criteria.
    It implements correct handling of distributed sampling in ``DataLoader``,
    so that the cuts are not duplicated across workers.

    Sampling in a ``CutSampler`` is intended to be very quick - it only uses the metadata in
    ``CutSet`` manifest to select the cuts, and is not intended to perform any I/O.

    CutSampler works similarly to PyTorch's DistributedSampler - when :attr:`shuffle=True`,
    you should call ``sampler.set_epoch(epoch)`` at each new epoch to have a different
    ordering of returned elements. However, its actual behaviour is different than that of
    DistributedSampler -- instead of partitioning the underlying cuts into equally sized chunks,
    it will return every N-th batch and skip the other batches (where ``N == world_size``).
    The formula used to determine which batches are returned is:
    ``(batch_idx + (world_size - rank)) % world_size == 0``.
    This ensures that we can return an equal number of batches in all distributed workers
    in spite of using a dynamic batch size, at the cost of skipping at most ``world_size - 1`` batches.

    Example usage::

        >>> dataset = K2SpeechRecognitionDataset(cuts)
        >>> sampler = SingleCutSampler(cuts, shuffle=True)
        >>> loader = DataLoader(dataset, sampler=sampler, batch_size=None)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     sampler.set_epoch(epoch)
        ...     train(loader)

    .. note::

        For implementers of new samplers:
        Subclasses of CutSampler are expected to implement ``self._next_batch()`` to introduce specific
        sampling logic (e.g. based on filters such as max number of frames/tokens/etc.).
        CutSampler defines ``__iter__()``, which optionally shuffles the cut IDs, and resets
        ``self.cut_idx`` to zero (to be used and incremented inside of ``_next_batch()``.
    """

    def __init__(
            self,
            shuffle: bool = False,
            world_size: Optional[int] = None,
            rank: Optional[int] = None,
            seed: int = 0,
            provide_len: bool = True
    ) -> None:
        """
        :param shuffle: When ``True``, the cuts will be shuffled at the start of iteration.
            Convenient when mini-batch loop is inside an outer epoch-level loop, e.g.:
            `for epoch in range(10): for batch in dataset: ...` as every epoch will see a
            different cuts order.
        :param world_size: Total number of distributed nodes. We will try to infer it by default.
        :param rank: Index of distributed node. We will try to infer it by default.
        :param seed: Random seed used to consistently shuffle the dataset across different processes.
        :param provide_len: Should we expose the ``__len__`` attribute in this class.
            It makes sense to turn it off when iterating the sampler is somewhat costly for any reason;
            e.g. because the underlying manifest is lazily loaded from the filesystem/somewhere else.
        """
        super().__init__(data_source=None)  # the "data_source" arg is not used in Sampler...
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.provide_len = provide_len
        if not self.provide_len and self.shuffle:
            warnings.warn("The sampler was set not to provide __len__, which suggests that you're "
                          "using lazy Cut manifests, but shuffle was set to True. "
                          "If your dataset is large, you might experience very slow performance "
                          "when iterating the sampler. To fix the issue, set shuffle=False.")

        self._maybe_init_distributed(world_size=world_size, rank=rank)
        self.num_batches = None
        self._filter_fn: Optional[Callable[[Cut], bool]] = None
        self.diagnostics = SamplingDiagnostics()

    def _maybe_init_distributed(self, world_size: Optional[int], rank: Optional[int]):
        if world_size is not None:
            assert world_size >= 1
        if rank is not None:
            assert rank >= 0
        if not dist.is_available() or not dist.is_initialized():
            self.world_size = 1 if world_size is None else world_size
            self.rank = 0 if rank is None else rank
            return
        self.world_size = dist.get_world_size() if world_size is None else world_size
        self.rank = dist.get_rank() if rank is None else rank
        assert self.rank < self.world_size

    def set_epoch(self, epoch: int) -> None:
        """
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        :param epoch: Epoch number.
        """
        self.epoch = epoch
        self.num_batches = None

    def filter(self, predicate: Callable[[Cut], bool]) -> None:
        """
        Add a constraint on invidual cuts that has to be satisfied to consider them.

        Can be useful when handling large, lazy manifests where it is not feasible to
        pre-filter them before instantiating the sampler.

        When set, we will remove the ``__len__`` attribute on the sampler, as it is now
        determined dynamically.

        Example:
            >>> cuts = CutSet(...)
            ... sampler = SingleCutSampler(cuts, max_duration=100.0)
            ... # Retain only the cuts that have at least 1s and at most 20s duration.
            ... sampler.filter(lambda cut: 1.0 <= cut.duration <= 20.0)
        """
        self._filter_fn = predicate
        self.provide_len = False

    def __iter__(self):
        raise NotImplementedError("Sub-classes of CutSampler have to implement __iter__()")

    def _next_batch(self):
        raise NotImplementedError("Sub-classes of CutSampler have to implement self._next_batch()")

    def __len__(self) -> int:
        if not self.provide_len:
            # Fake non-existence of this attribute
            raise TypeError(f"object of type '{type(self).__name__}' has no len()")
        if self.num_batches is None:
            self.num_batches = sum(1 for _ in self)
        return self.num_batches

    def __next__(self):
        # We use the following trick to ensure equal number of batches for each distributed
        # worker:
        # Every time a next batch is required, we will sample self.world_size batches first,
        # and then return the one at position self.rank.
        # This way, if any of the batches raises StopIteration, we'll know to stop early
        # when a given batch was available for one of the nodes, but not for the others.
        batches = []
        for _ in range(self.world_size):
            batches.append(self._next_batch())
        return batches[self.rank]

    def get_report(self) -> str:
        """Returns a string describing the statistics of the sampling process so far."""
        return self.diagnostics.get_report()


@dataclass
class TimeConstraint:
    """
    Represents a time-based constraint for sampler classes.
    It can be defined either as maximum total batch duration (in seconds),
    number of frames, or number of samples.
    These options are mutually exclusive and this class checks for that.

    :class:`TimeConstraint` can be used for tracking whether the criterion has been exceeded
    via the `add(cut)`, `exceeded()` and `reset()` methods.
    It will automatically track the right criterion (i.e. select frames/samples/duration from the cut).
    It can also be a null constraint (never exceeded).
    """
    max_duration: Optional[Seconds] = None
    max_samples: Optional[int] = None
    max_frames: Optional[int] = None
    current: Union[int, Seconds] = 0
    num_cuts: int = 0

    def __post_init__(self) -> None:
        assert exactly_one_not_null(*self._constraints) or all(x is None for x in self._constraints)
        for c in self._constraints:
            assert is_none_or_gt(c, 0)

    @property
    def _constraints(self) -> Tuple:
        return self.max_duration, self.max_frames, self.max_samples

    def is_active(self) -> bool:
        """Is it an actual constraint, or a dummy one (i.e. never exceeded)."""
        return any(x is not None for x in self._constraints)

    def add(self, cut: Cut) -> None:
        """
        Increment the internal counter for the time constraint,
        selecting the right property from the input ``cut`` object.
        """
        if self.max_frames is not None:
            self.current += cut.num_frames
        if self.max_samples is not None:
            self.current += cut.num_samples
        if self.max_duration is not None:
            self.current += cut.duration
        self.num_cuts += 1

    def exceeded(self) -> bool:
        """Is the constraint exceeded or not."""
        if self.max_frames is not None:
            return self.current > self.max_frames
        if self.max_samples is not None:
            return self.current > self.max_samples
        if self.max_duration is not None:
            return self.current > self.max_duration
        return False

    def close_to_exceeding(self) -> bool:
        """
        Check if the batch is close to satisfying the constraints.
        We define "closeness" as: if we added one more cut that has
        duration/num_frames/num_samples equal to the mean of the current
        batch, then the batch would have exceeded the constraints.
        """
        mean = self.current / self.num_cuts
        if self.max_frames is not None:
            return self.current + mean > self.max_frames
        if self.max_samples is not None:
            return self.current + mean > self.max_samples
        if self.max_duration is not None:
            return self.current + mean > self.max_duration
        return False

    def reset(self) -> None:
        """
        Reset the internal counter (to be used after a batch was created,
        to start collecting a new one).
        """
        self.current = 0
        self.num_cuts = 0

    def __add__(self, other: 'TimeConstraint') -> 'TimeConstraint':
        for key in ('max_duration', 'max_frames', 'max_samples'):
            self_attr = getattr(self, key)
            other_attr = getattr(other, key)
            is_none = self_attr is None and other_attr is None
            assert is_none or isclose(self_attr, other_attr), (
                f"To add two TimeConstraint objects, they need to represent the same constraint "
                f"(got self.{key}={self_attr} != other.{key}={other_attr})."
            )
        return TimeConstraint(
            max_duration=self.max_duration,
            max_frames=self.max_frames,
            max_samples=self.max_samples,
            current=self.current + other.current,
            num_cuts=self.num_cuts + other.num_cuts
        )


@dataclass
class SamplingDiagnostics:
    """
    Utility for collecting diagnostics about the sampling process:
    how many cuts/batches were discarded.
    """
    kept_stats: TimeConstraint = field(default_factory=lambda: TimeConstraint(max_duration=float('inf')))
    discarded_stats: TimeConstraint = field(default_factory=lambda: TimeConstraint(max_duration=float('inf')))
    num_kept_batches: int = 0
    num_discarded_batches: int = 0

    def keep(self, cuts: Iterable[Cut]) -> None:
        cntr = 0
        for cut in cuts:
            self.kept_stats.add(cut)
            cntr += 1
        if not cntr:
            warnings.warn('Found an accepted batch with zero cuts. This could be an error.')
        self.num_kept_batches += 1

    def discard(self, cuts: Iterable[Cut]) -> None:
        cntr = 0
        for cut in cuts:
            self.discarded_stats.add(cut)
            cntr += 1
        if cntr:
            # We don't warn about discarded batches with 0 cuts.
            self.num_discarded_batches += 1

    def reset(self) -> None:
        self.kept_stats.reset()
        self.discarded_stats.reset()
        self.num_kept_batches = 0
        self.num_discarded_batches = 0

    @property
    def total_cuts(self) -> int:
        return self.kept_stats.num_cuts + self.discarded_stats.num_cuts

    @property
    def total_batches(self) -> int:
        return self.num_kept_batches + self.num_discarded_batches

    def get_report(self) -> str:
        """Returns a string describing the statistics of the sampling process so far."""
        if self.total_batches == 0 or self.total_cuts == 0:
            return "Sampling statistics unvavailable: the SamplerDiagnostics received no cuts or batches. " \
                   "If this is unexpected, and you're using a custom sampler, ensure that the sampler " \
                   "is registering the batches in SamplerDiagnostics."
        return (
            f"Sampling statistics: \n"
            f"Kept {self.kept_stats.num_cuts:d}/{self.total_cuts:d} "
            f"({self.kept_stats.num_cuts / self.total_cuts:.2%}) cuts "
            f"({self.discarded_stats.num_cuts:d} cuts discarded).\n"
            f"Kept {self.num_kept_batches:d}/{self.total_batches:d} "
            f"({self.num_kept_batches / self.total_batches:.2%}) batches "
            f"({self.num_discarded_batches:d} batches discarded).\n"
            f"Overall, {round(self.discarded_stats.current):d} seconds of supervision were discarded."
        )

    def __add__(self, other: 'SamplingDiagnostics') -> 'SamplingDiagnostics':
        return SamplingDiagnostics(
            kept_stats=self.kept_stats + other.kept_stats,
            discarded_stats=self.discarded_stats + other.discarded_stats,
            num_kept_batches=self.num_kept_batches + other.num_kept_batches,
            num_discarded_batches=self.num_discarded_batches + other.num_discarded_batches
        )


class SingleCutSampler(CutSampler):
    """
    Samples cuts from a CutSet to satisfy the input constraints.
    It behaves like an iterable that yields lists of strings (cut IDs).

    When one of :attr:`max_frames`, :attr:`max_samples`, or :attr:`max_duration` is specified,
    the batch size is dynamic.
    Exactly zero or one of those constraints can be specified.
    Padding required to collate the batch does not contribute to max frames/samples/duration.

    Example usage::

        >>> dataset = K2SpeechRecognitionDataset(cuts)
        >>> sampler = SingleCutSampler(cuts, shuffle=True)
        >>> loader = DataLoader(dataset, sampler=sampler, batch_size=None)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     sampler.set_epoch(epoch)
        ...     train(loader)

    """

    def __init__(
            self,
            cuts: CutSet,
            max_frames: int = None,
            max_samples: int = None,
            max_duration: Seconds = None,
            max_cuts: Optional[int] = None,
            shuffle: bool = False,
            drop_last: bool = False,
            world_size: Optional[int] = None,
            rank: Optional[int] = None,
            seed: int = 0,
    ):
        """
        SingleCutSampler's constructor.

        :param cuts: the ``CutSet`` to sample data from.
        :param max_frames: The maximum total number of feature frames from ``cuts``.
        :param max_samples: The maximum total number of audio samples from ``cuts``.
        :param max_duration: The maximum total recording duration from ``cuts``.
        :param max_cuts: The maximum number of cuts sampled to form a mini-batch.
            By default, this constraint is off.
        :param shuffle: When ``True``, the cuts will be shuffled at the start of iteration.
            Convenient when mini-batch loop is inside an outer epoch-level loop, e.g.:
            `for epoch in range(10): for batch in dataset: ...` as every epoch will see a
            different cuts order.
        :param drop_last: When ``True``, the last batch is dropped if it's incomplete.
        :param world_size: Total number of distributed nodes. We will try to infer it by default.
        :param rank: Index of distributed node. We will try to infer it by default.
        :param seed: Random seed used to consistently shuffle the dataset across different processes.
        """
        super().__init__(
            provide_len=not cuts.is_lazy,
            shuffle=shuffle,
            world_size=world_size,
            rank=rank,
            seed=seed,
        )
        self.data_source = DataSource(cuts)
        self.time_constraint = TimeConstraint(
            max_duration=max_duration,
            max_frames=max_frames,
            max_samples=max_samples
        )
        self.drop_last = drop_last
        self.max_cuts = max_cuts
        assert self.time_constraint.is_active() or \
               not (self.time_constraint.is_active() and self.max_cuts is not None)
        # Constraints
        assert is_none_or_gt(self.max_cuts, 0)

    def __iter__(self) -> 'SingleCutSampler':
        """
        Prepare the dataset for iterating over a new epoch. Will shuffle the data if requested.
        """
        if self.shuffle:
            self.data_source.shuffle(self.seed + self.epoch)
        iter(self.data_source)
        self.diagnostics.reset()
        return self

    def _next_batch(self) -> CutSet:
        # Keep iterating the underlying CutSet as long as we hit or exceed the constraints
        # provided by user (the max number of frames or max number of cuts).
        # Note: no actual data is loaded into memory yet because the manifests contain all the metadata
        # required to do this operation.
        self.time_constraint.reset()
        cuts = []
        while True:

            # Check that we have not reached the end of the dataset.
            try:
                # If this doesn't raise (typical case), it's not the end: keep processing.
                next_cut = next(self.data_source)
            except StopIteration:
                # No more cuts to sample from: if we have a partial batch,
                # we may output it, unless the user requested to drop it.
                # We also check if the batch is "almost there" to override drop_last.
                if cuts and (not self.drop_last or self.time_constraint.close_to_exceeding()):
                    # We have a partial batch and we can return it.
                    self.diagnostics.keep(cuts)
                    return CutSet.from_cuts(cuts)
                else:
                    # There is nothing more to return or it's discarded:
                    # signal the iteration code to stop.
                    self.diagnostics.discard(cuts)
                    raise StopIteration()

            # Check whether the cut we're about to sample satisfies optional user-requested predicate.
            if self._filter_fn is not None and not self._filter_fn(next_cut):
                # No - try another one.
                continue

            # Track the duration/frames/etc. constraints.
            self.time_constraint.add(next_cut)
            next_num_cuts = len(cuts) + 1

            # Did we exceed the max_frames and max_cuts constraints?
            if not self.time_constraint.exceeded() and (self.max_cuts is None or next_num_cuts <= self.max_cuts):
                # No - add the next cut to the batch, and keep trying.
                cuts.append(next_cut)
            else:
                # Yes. Do we have at least one cut in the batch?
                if cuts:
                    # Yes. Return the batch, but keep the currently drawn cut for later.
                    self.data_source.take_back(next_cut)
                    break
                else:
                    # No. We'll warn the user that the constrains might be too tight,
                    # and return the cut anyway.
                    warnings.warn("The first cut drawn in batch collection violates the max_frames or max_cuts "
                                  "constraints - we'll return it anyway. Consider increasing max_frames/max_cuts.")
                    cuts.append(next_cut)

        self.diagnostics.keep(cuts)
        return CutSet.from_cuts(cuts)


class CutPairsSampler(CutSampler):
    """
    Samples pairs of cuts from a "source" and "target" CutSet.
    It expects that both CutSet's strictly consist of Cuts with corresponding IDs.
    It behaves like an iterable that yields lists of strings (cut IDs).

    When one of :attr:`max_frames`, :attr:`max_samples`, or :attr:`max_duration` is specified,
    the batch size is dynamic.
    Exactly zero or one of those constraints can be specified.
    Padding required to collate the batch does not contribute to max frames/samples/duration.
    """

    def __init__(
            self,
            source_cuts: CutSet,
            target_cuts: CutSet,
            max_source_frames: int = None,
            max_source_samples: int = None,
            max_source_duration: Seconds = None,
            max_target_frames: int = None,
            max_target_samples: int = None,
            max_target_duration: int = None,
            max_cuts: Optional[int] = None,
            shuffle: bool = False,
            drop_last: bool = False,
            world_size: Optional[int] = None,
            rank: Optional[int] = None,
            seed: int = 0,
    ):
        """
        CutPairsSampler's constructor.

        :param source_cuts: the first ``CutSet`` to sample data from.
        :param target_cuts: the second ``CutSet`` to sample data from.
        :param max_source_frames: The maximum total number of feature frames from ``source_cuts``.
        :param max_source_samples: The maximum total number of audio samples from ``source_cuts``.
        :param max_source_duration: The maximum total recording duration from ``source_cuts``.
        :param max_target_frames: The maximum total number of feature frames from ``target_cuts``.
        :param max_target_samples: The maximum total number of audio samples from ``target_cuts``.
        :param max_target_duration: The maximum total recording duration from ``target_cuts``.
        :param max_cuts: The maximum number of cuts sampled to form a mini-batch.
            By default, this constraint is off.
        :param shuffle: When ``True``, the cuts will be shuffled at the start of iteration.
            Convenient when mini-batch loop is inside an outer epoch-level loop, e.g.:
            `for epoch in range(10): for batch in dataset: ...` as every epoch will see a
            different cuts order.
        :param drop_last: When ``True``, the last batch is dropped if it's incomplete.
        :param world_size: Total number of distributed nodes. We will try to infer it by default.
        :param rank: Index of distributed node. We will try to infer it by default.
        :param seed: Random seed used to consistently shuffle the dataset across different processes.
        """
        super().__init__(
            provide_len=not source_cuts.is_lazy and not target_cuts.is_lazy,
            shuffle=shuffle,
            world_size=world_size,
            rank=rank,
            seed=seed,
        )
        self.source_cuts = DataSource(source_cuts)
        self.target_cuts = DataSource(target_cuts)
        # Constraints
        self.source_constraints = TimeConstraint(
            max_duration=max_source_duration,
            max_samples=max_source_samples,
            max_frames=max_source_frames
        )
        self.target_constraints = TimeConstraint(
            max_duration=max_target_duration,
            max_samples=max_target_samples,
            max_frames=max_target_frames
        )
        self.max_cuts = max_cuts
        self.drop_last = drop_last

    def __iter__(self) -> 'CutPairsSampler':
        """
        Prepare the dataset for iterating over a new epoch. Will shuffle the data if requested.
        """
        if self.shuffle:
            self.source_cuts.shuffle(self.seed + self.epoch)
            self.target_cuts.sort_like(self.source_cuts)
        iter(self.source_cuts)
        iter(self.target_cuts)
        self.diagnostics.reset()
        return self

    def _next_batch(self) -> Tuple[CutSet, CutSet]:
        # Keep iterating the underlying CutSets as long as we hit or exceed the constraints
        # provided by user (the max number of source_feats or max number of cuts).
        # Note: no actual data is loaded into memory yet because the manifests contain all the metadata
        # required to do this operation.
        self.source_constraints.reset()
        self.target_constraints.reset()
        source_cuts = []
        target_cuts = []
        while True:
            # Check that we have not reached the end of the dataset.
            try:
                # We didn't - grab the next cut
                next_source_cut = next(self.source_cuts)
                next_target_cut = next(self.target_cuts)
                assert next_source_cut.id == next_target_cut.id, (
                    "Sampled source and target cuts with differing IDs. "
                    "Ensure that your source and target cuts have the same length, "
                    "the same IDs, and the same order."
                )
            except StopIteration:
                # No more cuts to sample from: if we have a partial batch,
                # we may output it, unless the user requested to drop it.
                # We also check if the batch is "almost there" to override drop_last.
                if source_cuts and (
                        not self.drop_last
                        or self.source_constraints.close_to_exceeding()
                        or self.target_constraints.close_to_exceeding()
                ):
                    # We have a partial batch and we can return it.
                    assert len(source_cuts) == len(
                        target_cuts
                    ), "Unexpected state: some cuts in source / target are missing their counterparts..."
                    self.diagnostics.keep(source_cuts)
                    return CutSet.from_cuts(source_cuts), CutSet.from_cuts(target_cuts)
                else:
                    # There is nothing more to return or it's discarded:
                    # signal the iteration code to stop.
                    self.diagnostics.discard(source_cuts)
                    raise StopIteration()

            # Check whether the cuts we're about to sample satisfy optional user-requested predicate.
            if self._filter_fn is not None and (
                    not self._filter_fn(next_source_cut)
                    or not self._filter_fn(next_target_cut)
            ):
                # No - try another one.
                continue

            self.source_constraints.add(next_source_cut)
            self.target_constraints.add(next_target_cut)
            next_num_cuts = len(source_cuts) + 1

            # Did we exceed the max_source_frames and max_cuts constraints?
            if not self.source_constraints.exceeded() \
                    and not self.target_constraints.exceeded() \
                    and (self.max_cuts is None or next_num_cuts <= self.max_cuts):
                # No - add the next cut to the batch, and keep trying.
                source_cuts.append(next_source_cut)
                target_cuts.append(next_target_cut)
            else:
                # Yes. Do we have at least one cut in the batch?
                if source_cuts:
                    # Yes. Return it.
                    self.source_cuts.take_back(next_source_cut)
                    self.target_cuts.take_back(next_target_cut)
                    break
                else:
                    # No. We'll warn the user that the constrains might be too tight,
                    # and return the cut anyway.
                    warnings.warn("The first cut drawn in batch collection violates one of the max_... constraints"
                                  "we'll return it anyway. Consider increasing max_source_frames/max_cuts/etc.")
                    source_cuts.append(next_source_cut)
                    target_cuts.append(next_target_cut)

        assert len(source_cuts) == len(
            target_cuts
        ), "Unexpected state: some cuts in source / target are missing their counterparts..."
        self.diagnostics.keep(source_cuts)
        return CutSet.from_cuts(source_cuts), CutSet.from_cuts(target_cuts)


class BucketingSampler(CutSampler):
    """
    Sorts the cuts in a :class:`CutSet` by their duration and puts them into similar duration buckets.
    For each bucket, it instantiates a simpler sampler instance, e.g. :class:`SingleCutSampler`.

    It behaves like an iterable that yields lists of strings (cut IDs).
    During iteration, it randomly selects one of the buckets to yield the batch from,
    until all the underlying samplers are depleted (which means it's the end of an epoch).

    Examples:

    Bucketing sampler with 20 buckets, sampling single cuts::

        >>> sampler = BucketingSampler(
        ...    cuts,
        ...    # BucketingSampler specific args
        ...    sampler_type=SingleCutSampler, num_buckets=20,
        ...    # Args passed into SingleCutSampler
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
            sampler_type: Type = SingleCutSampler,
            num_buckets: int = 10,
            drop_last: bool = False,
            seed: int = 0,
            **kwargs
    ):
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
            provide_len=all(not cs.is_lazy for cs in cuts),
            world_size=1,
            rank=0,
            seed=seed
        )
        self.num_buckets = num_buckets
        self.drop_last = drop_last
        self.sampler_type = sampler_type
        self.sampler_kwargs = kwargs
        self.cut_sets = cuts
        first_cut_set = cuts[0].sort_by_duration()
        buckets = [
            cs.sort_like(first_cut_set).split(num_buckets) for cs in self.cut_sets
        ]
        # zip(*buckets) does:
        # [(cs0_0, cs1_0, cs2_0), (cs0_1, cs1_1, cs2_1)] -> [(cs0_0, cs0_1), (cs1_0, cs1_1), (cs2_0, cs2_1)]
        self.buckets = list(zip(*buckets))
        self.bucket_samplers = [
            self.sampler_type(*bucket_cut_sets, drop_last=drop_last, **kwargs)
            for bucket_cut_sets in self.buckets
        ]
        self.bucket_rng = random.Random(self.seed + self.epoch)
        self.depleted = [False] * num_buckets

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
        Add a constraint on invidual cuts that has to be satisfied to consider them.

        Can be useful when handling large, lazy manifests where it is not feasible to
        pre-filter them before instantiating the sampler.

        When set, we will remove the ``__len__`` attribute on the sampler, as it is now
        determined dynamically.

        Example:
            >>> cuts = CutSet(...)
            ... sampler = SingleCutSampler(cuts, max_duration=100.0)
            ... # Retain only the cuts that have at least 1s and at most 20s duration.
            ... sampler.filter(lambda cut: 1.0 <= cut.duration <= 20.0)
        """
        for sampler in self.bucket_samplers:
            sampler.filter(predicate)

    def __iter__(self) -> 'BucketingSampler':
        self.bucket_rng.seed(self.seed + self.epoch)
        for b in self.bucket_samplers:
            iter(b)
        self.depleted = [False] * self.num_buckets
        return self

    def _next_batch(self):
        while not self.is_depleted:
            idx, sampler = self.bucket_rng.choice(self._nondepleted_samplers_with_idxs)
            try:
                return next(sampler)
            except StopIteration:
                self.depleted[idx] = True
        raise StopIteration()

    def __len__(self):
        if self.num_batches is None:
            self.num_batches = sum(
                len(sampler)
                for sampler in self.bucket_samplers
            )
        return self.num_batches

    @property
    def is_depleted(self) -> bool:
        return all(self.depleted)

    @property
    def _nondepleted_samplers_with_idxs(self):
        return [
            (idx, bs) for idx, (bs, depleted) in
            enumerate(zip(self.bucket_samplers, self.depleted))
            if not depleted
        ]

    def get_report(self) -> str:
        """Returns a string describing the statistics of the sampling process so far."""
        total_diagnostics = reduce(add, (bucket.diagnostics for bucket in self.bucket_samplers))
        return total_diagnostics.get_report()


class ZipSampler(CutSampler):
    """
    :class:`.ZipSampler` takes several samplers as input and concatenates their
    sampled batch cuts together into a single list.
    It is helpful for ensuring that each batch consists of some proportion of cuts
    coming from different sources.

    The input samplers do not have to provide the same number of batches -- when
    any of the samplers becomes depleted, the iteration will stop (like with
    Python's ``zip()`` function).

    Example::

        >>> sampler = ZipSampler(
        ...     SingleCutSampler(cuts_corpusA, max_duration=250, shuffle=True),
        ...     SingleCutSampler(cuts_corpusB, max_duration=100, shuffle=True),
        ... )
        >>> for cut in sampler:
        ...     pass  # profit
    """
    def __init__(self, *samplers: CutSampler) -> None:
        super().__init__([])  # dummy initialization, might need to refactor.
        self.samplers = samplers

    def __iter__(self):
        for sampler in self.samplers:
            iter(sampler)
        return self

    def _next_batch(self) -> List[str]:
        batches = []
        for sampler in self.samplers:
            batches.append(next(sampler))
        return [item for batch in batches for item in batch]

    def set_epoch(self, epoch: int) -> None:
        """
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        :param epoch: Epoch number.
        """
        for s in self.samplers:
            s.set_epoch(epoch)
        super().set_epoch(epoch)

    def filter(self, predicate: Callable[[Cut], bool]) -> None:
        """
        Add a constraint on invidual cuts that has to be satisfied to consider them.

        Can be useful when handling large, lazy manifests where it is not feasible to
        pre-filter them before instantiating the sampler.

        When set, we will remove the ``__len__`` attribute on the sampler, as it is now
        determined dynamically.

        Example:
            >>> cuts = CutSet(...)
            ... sampler = SingleCutSampler(cuts, max_duration=100.0)
            ... # Retain only the cuts that have at least 1s and at most 20s duration.
            ... sampler.filter(lambda cut: 1.0 <= cut.duration <= 20.0)
        """
        for sampler in self.samplers:
            sampler.filter(predicate)

    def __len__(self):
        if self.num_batches is None:
            self.num_batches = min(len(sampler) for sampler in self.samplers)
        return self.num_batches
      
    def get_report(self) -> str:
        """Returns a string describing the statistics of the sampling process so far."""
        total_diagnostics = reduce(add, (sampler.diagnostics for sampler in self.samplers))
        return total_diagnostics.get_report()


def find_pessimistic_batches(
        sampler: CutSampler, batch_tuple_index: int = 0
) -> Tuple[Dict[str, CutSet], Dict[str, float]]:
    """
    Function for finding 'pessimistic' batches, i.e. batches that have the highest potential
    to blow up the GPU memory during training. We will fully iterate the sampler and record
    the most risky batches under several criteria:
    - single longest cut
    - single longest supervision
    - largest batch cuts duration
    - largest batch supervisions duration
    - max num cuts
    - max num supervisions

    .. note: It is up to the users to convert the sampled CutSets into actual batches and test them
        by running forward and backward passes with their model.

    Example of how this function can be used with a PyTorch model
    and a :class:`~lhotse.dataset.K2SpeechRecognitionDataset`::

        sampler = SingleCutSampler(cuts, max_duration=300)
        dataset = K2SpeechRecognitionDataset()
        batches, scores = find_pessimistic_batches(sampler)
        for reason, cuts in batches.items():
            try:
                batch = dset[cuts]
                outputs = model(batch)
                loss = loss_fn(outputs)
                loss.backward()
            except:
                print(f"Exception caught when evaluating pessimistic batch for: {reason}={scores[reason]}")
                raise


    :param sampler: An instance of a Lhotse :class:`.CutSampler`.
    :param batch_tuple_index: Applicable to samplers that return tuples of :class:`~lhotse.cut.CutSet`.
        Indicates which position in the tuple we should look up for the CutSet.
    :return: A tuple of dicts: the first with batches (as CutSets) and the other with criteria values, i.e.:
        ``({"<criterion>": <CutSet>, ...}, {"<criterion>": <value>, ...})``
    """
    criteria = {
        "single_longest_cut": lambda cuts: max(c.duration for c in cuts),
        "single_longest_supervision": lambda cuts: max(
            sum(s.duration for s in c.supervisions) for c in cuts
        ),
        "largest_batch_cuts_duration": lambda cuts: sum(c.duration for c in cuts),
        "largest_batch_supervisions_duration": lambda cuts: sum(
            s.duration for c in cuts for s in c.supervisions
        ),
        "max_num_cuts": len,
        "max_num_supervisions": lambda cuts: sum(
            1 for c in cuts for _ in c.supervisions
        ),
    }
    try:
        sampler = iter(sampler)
        first_batch = next(sampler)
        if isinstance(first_batch, tuple):
            first_batch = first_batch[batch_tuple_index]
    except StopIteration:
        warnings.warn("Empty sampler encountered in find_pessimistic_batches()")
        return {}

    top_batches = {k: first_batch for k in criteria}
    top_values = {k: fn(first_batch) for k, fn in criteria.items()}

    for batch in sampler:
        if isinstance(batch, tuple):
            batch = batch[batch_tuple_index]
        for crit, fn in criteria.items():
            val = fn(batch)
            if val > top_values[crit]:
                top_values[crit] = val
                top_batches[crit] = batch

    return top_batches, top_values
