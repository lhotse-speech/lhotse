import random
import warnings
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple, Type, Union

import torch.distributed as dist
from torch.utils.data import Sampler

from lhotse import CutSet
from lhotse.cut import Cut
from lhotse.utils import Seconds, exactly_one_not_null, is_none_or_gt


class DataSource:
    def __init__(self, items):
        self._items = items
        self._permutation = list(range(len(self._items)))

    def shuffle(self, seed):
        r = random.Random(seed)
        self._permutation = list(range(len(self._items)))
        r.shuffle(self._permutation)

    def __getitem__(self, idx):
        return self._items[self._permutation[idx]]

    def __len__(self):
        return len(self._items)


class CutSampler(Sampler[List[str]]):
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
            cut_ids: Iterable[str],
            shuffle: bool = False,
            world_size: Optional[int] = None,
            rank: Optional[int] = None,
            seed: int = 0,
            provide_len: bool = True
    ) -> None:
        """

        :param cut_ids: An iterable of cut IDs for the full dataset.
            CutSampler will take care of partitioning that into distributed workers (if needed).
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
        data_source = DataSource(list(cut_ids))
        super().__init__(data_source)
        self.data_source = data_source
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.cut_idx = 0
        self.provide_len = provide_len
        if not self.provide_len and self.shuffle:
            warnings.warn("The sampler was set not to provide __len__, which suggests that you're "
                          "using lazy Cut manifests, but shuffle was set to True. "
                          "If your dataset is large, you might experience very slow performance "
                          "when iterating the sampler. To fix the issue, set shuffle=False.")

        self._maybe_init_distributed(world_size=world_size, rank=rank)
        self.num_batches = None
        self._filter_fn: Optional[Callable[[Cut], bool]] = None

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

    def _next_batch(self) -> List[str]:
        raise NotImplementedError("Sub-classes of CutSampler have to implement self._next_batch()")

    def __iter__(self) -> 'CutSampler':
        """
        Prepare the dataset for iterating over a new epoch. Will shuffle the data if requested.
        """
        if self.shuffle:
            self.data_source.shuffle(self.seed + self.epoch)
        self.cut_idx = 0
        return self

    def __len__(self) -> int:
        if not self.provide_len:
            # Fake non-existence of this attribute
            raise TypeError(f"object of type '{type(self).__name__}' has no len()")
        if self.num_batches is None:
            self.num_batches = sum(1 for _ in self)
        return self.num_batches

    def __next__(self) -> List[str]:
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

    def exceeded(self) -> bool:
        """Is the constraint exceeded or not."""
        if self.max_frames is not None:
            return self.current > self.max_frames
        if self.max_samples is not None:
            return self.current > self.max_samples
        if self.max_duration is not None:
            return self.current > self.max_duration
        return False

    def reset(self) -> None:
        """
        Reset the internal counter (to be used after a batch was created,
        to start collecting a new one).
        """
        self.current = 0


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
            **kwargs
    ):
        """
        SingleCutSampler's constructor.

        :param cuts: the ``CutSet`` to sample data from.
        :param max_frames: The maximum total number of feature frames from ``cuts``.
        :param max_samples: The maximum total number of audio samples from ``cuts``.
        :param max_duration: The maximum total recording duration from ``cuts``.
        :param max_cuts: The maximum number of cuts sampled to form a mini-batch.
            By default, this constraint is off.
        :param kwargs: Arguments to be passed into ``CutSampler``.
        """
        super().__init__(cuts.ids, provide_len=not cuts.is_lazy, **kwargs)
        self.cuts = cuts
        self.time_constraint = TimeConstraint(
            max_duration=max_duration,
            max_frames=max_frames,
            max_samples=max_samples
        )
        self.max_cuts = max_cuts
        assert self.time_constraint.is_active() or \
               not (self.time_constraint.is_active() and self.max_cuts is not None)
        # Constraints
        assert is_none_or_gt(self.max_cuts, 0)

    def _next_batch(self) -> List[str]:
        # Keep iterating the underlying CutSet as long as we hit or exceed the constraints
        # provided by user (the max number of frames or max number of cuts).
        # Note: no actual data is loaded into memory yet because the manifests contain all the metadata
        # required to do this operation.
        self.time_constraint.reset()
        cut_ids = []
        while True:
            # Check that we have not reached the end of the dataset.
            if self.cut_idx < len(self.data_source):
                # We didn't - grab the next cut
                next_cut_id = self.data_source[self.cut_idx]
            else:
                if cut_ids:
                    # We did and we have a partial batch - return it.
                    return cut_ids
                else:
                    # We did and there is nothing more to return - signal the iteration code to stop.
                    raise StopIteration()
            next_cut = self.cuts[next_cut_id]
            # Check whether the cut we're about to sample satisfies optional user-requested predicate.
            if self._filter_fn is not None and not self._filter_fn(next_cut):
                # No - try another one.
                self.cut_idx += 1
                continue
            self.time_constraint.add(next_cut)
            next_num_cuts = len(cut_ids) + 1
            # Did we exceed the max_frames and max_cuts constraints?
            if not self.time_constraint.exceeded() and (self.max_cuts is None or next_num_cuts <= self.max_cuts):
                # No - add the next cut to the batch, and keep trying.
                cut_ids.append(next_cut.id)
                self.cut_idx += 1
            else:
                # Yes. Do we have at least one cut in the batch?
                if cut_ids:
                    # Yes. Return it.
                    break
                else:
                    # No. We'll warn the user that the constrains might be too tight,
                    # and return the cut anyway.
                    warnings.warn("The first cut drawn in batch collection violates the max_frames or max_cuts "
                                  "constraints - we'll return it anyway. Consider increasing max_frames/max_cuts.")
                    cut_ids.append(next_cut.id)
                    self.cut_idx += 1
        return cut_ids


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
            **kwargs
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
        """
        super().__init__(
            source_cuts.ids,
            provide_len=not source_cuts.is_lazy and not target_cuts.is_lazy,
            **kwargs
        )
        self.source_cuts = source_cuts
        self.target_cuts = target_cuts
        assert set(self.source_cuts.ids) == set(self.target_cuts.ids), \
            "Expected source and target cuts to have the same set of IDs."
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

    def _next_batch(self) -> List[str]:
        # Keep iterating the underlying CutSets as long as we hit or exceed the constraints
        # provided by user (the max number of source_feats or max number of cuts).
        # Note: no actual data is loaded into memory yet because the manifests contain all the metadata
        # required to do this operation.
        self.source_constraints.reset()
        self.target_constraints.reset()
        cut_ids = []
        while True:
            # Check that we have not reached the end of the dataset.
            if self.cut_idx < len(self.data_source):
                # We didn't - grab the next cut
                next_cut_id = self.data_source[self.cut_idx]
            else:
                if cut_ids:
                    # We did and we have a partial batch - return it.
                    return cut_ids
                else:
                    # We did and there is nothing more to return - signal the iteration code to stop.
                    raise StopIteration()
            next_source_cut = self.source_cuts[next_cut_id]
            next_target_cut = self.target_cuts[next_cut_id]
            # Check whether the cuts we're about to sample satisfy optional user-requested predicate.
            if self._filter_fn is not None and (
                    not self._filter_fn(next_source_cut)
                    or not self._filter_fn(next_target_cut)
            ):
                # No - try another one.
                self.cut_idx += 1
                continue
            self.source_constraints.add(next_source_cut)
            self.target_constraints.add(next_target_cut)
            next_num_cuts = len(cut_ids) + 1
            # Did we exceed the max_source_frames and max_cuts constraints?
            if not self.source_constraints.exceeded() \
                    and not self.target_constraints.exceeded() \
                    and (self.max_cuts is None or next_num_cuts <= self.max_cuts):
                # No - add the next cut to the batch, and keep trying.
                cut_ids.append(next_source_cut.id)
                self.cut_idx += 1
            else:
                # Yes. Do we have at least one cut in the batch?
                if cut_ids:
                    # Yes. Return it.
                    break
                else:
                    # No. We'll warn the user that the constrains might be too tight,
                    # and return the cut anyway.
                    warnings.warn("The first cut drawn in batch collection violates one of the max_... constraints"
                                  "we'll return it anyway. Consider increasing max_source_frames/max_cuts/etc.")
                    cut_ids.append(next_source_cut.id)
                    self.cut_idx += 1
        return cut_ids


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
        :param seed: random seed for bucket selection
        :param kwargs: Arguments used to create the underlying sampler for each bucket.
        """
        # Do not use the distributed capacities of the CutSampler in the top-level sampler.
        super().__init__(
            cuts[0].ids,
            provide_len=all(not cs.is_lazy for cs in cuts),
            world_size=1,
            rank=0,
            seed=seed
        )
        self.num_buckets = num_buckets
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
            self.sampler_type(*bucket_cut_sets, **kwargs)
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

    def _next_batch(self) -> List[str]:
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
