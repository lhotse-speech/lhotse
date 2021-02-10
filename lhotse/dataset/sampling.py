import logging
import random
import warnings
from math import ceil
from typing import Iterable, List, Optional, Type

import torch.distributed as dist
from torch.utils.data import Sampler

from lhotse import CutSet


class CutSampler(Sampler[List[str]]):
    """
    CutSampler is responsible for collecting batches of cuts, given specified criteria.
    It implements correct handling of distributed sampling in DataLoader,
    so that the cuts are not duplicated across workers.

    Sampling in a CutSampler is intended to be very quick - it only uses the metadata in
    ``CutSet`` manifest to select the cuts, and is not intended to perform any I/O.

    CutSampler works similarly to PyTorch's DistributedSampler - when :attr:`shuffle=True`,
    you should call ``sampler.set_epoch(epoch)`` at each new epoch to have a different
    ordering of returned elements.

    Example usage::

        >>> dataset = K2SpeechRecognitionDataset(cuts)
        >>> sampler = SingleCutSampler(cuts, shuffle=True)
        >>> loader = DataLoader(dataset, sampler=sampler, batch_size=None)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     sampler.set_epoch(epoch)
        ...     train(loader)

    .. note::

        For implementers of new samplers:
        Subclasses of CutSampler are expected to implement ``__next__()`` to introduce specific
        sampling logic (e.g. based on filters such as max number of frames/tokens/etc.).
        CutSampler defines ``__iter__()``, which optionally shuffles the cut IDs, and resets
        ``self.current_idx`` to zero (to be used and incremented inside of ``__next__()``.
    """

    def __init__(
            self,
            cut_ids: Iterable[str],
            shuffle: bool = False,
            world_size: Optional[int] = None,
            rank: Optional[int] = None,
            seed: int = 0,
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
        """
        data_source = list(cut_ids)
        super().__init__(data_source)
        self.full_data_source = data_source
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self._maybe_init_distributed(world_size=world_size, rank=rank)
        self.data_source = partition_cut_ids(self.full_data_source, world_size=self.world_size, rank=self.rank)

    def _maybe_init_distributed(self, world_size: Optional[int], rank: Optional[int]):
        try:
            # We will ask PyTorch about distributed training metadata,
            # and it might blow in our faces.
            if world_size is None:
                if not dist.is_available():
                    raise RuntimeError("Requires distributed package to be available")
                self.world_size = dist.get_world_size()
            else:
                self.world_size = world_size
            if rank is None:
                if not dist.is_available():
                    raise RuntimeError("Requires distributed package to be available")
                self.rank = dist.get_rank()
            else:
                self.rank = rank
        except AssertionError as e:
            if 'process group is not initialized' in str(e):
                # Distributed training not active OR somebody forgot to initialize the process group
                # (which will come up anyway at some point, so we can ignore it here).
                self.world_size = 1
                self.rank = 0
            else:
                # A different error - pass it on.
                raise

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        :param epoch: Epoch number.
        """
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.data_source)

    def __iter__(self) -> 'CutSampler':
        """
        Prepare the dataset for iterating over a new epoch. Will shuffle the data if requested.
        """
        if self.shuffle:
            r = random.Random(self.seed + self.epoch)
            r.shuffle(self.data_source)
        self.current_idx = 0
        return self

    def __next__(self) -> List[str]:
        raise NotImplemented


class SingleCutSampler(CutSampler):
    """
    Samples cuts from a CutSet to satisfy the criteria of max_frames and max_cuts.
    It behaves like an iterable that yields lists of strings (cut IDs).
    """

    def __init__(
            self,
            cuts: CutSet,
            max_frames: int = 26000,
            max_cuts: Optional[int] = None,
            **kwargs
    ):
        """
        SingleCutSampler's constructor.

        :param cuts: the ``CutSet`` to sample data from.
        :param max_frames: The maximum number of feature frames from ``cuts``
            that we're going to put in a single batch.
            The padding introduced during collation does not contribute to that limit.
        :param max_cuts: The maximum number of cuts sampled to form a mini-batch.
            By default, this constraint is off.
        :param kwargs: Arguments to be passed into ``CutSampler``.
        """
        super().__init__(cuts.ids, **kwargs)
        self.cuts = cuts
        # Constraints
        self.max_frames = max_frames
        self.max_cuts = max_cuts
        assert self.max_frames > 0
        assert self.max_cuts is None or self.max_cuts > 0

    def __next__(self) -> List[str]:
        # Keep iterating the underlying CutSet as long as we hit or exceed the constraints
        # provided by user (the max number of frames or max number of cuts).
        # Note: no actual data is loaded into memory yet because the manifests contain all the metadata
        # required to do this operation.
        num_frames = 0
        cut_ids = []
        while True:
            # Check that we have not reached the end of the dataset.
            if self.current_idx < len(self.data_source):
                # We didn't - grab the next cut
                next_cut_id = self.data_source[self.current_idx]
            else:
                if cut_ids:
                    # We did and we have a partial batch - return it.
                    return cut_ids
                else:
                    # We did and there is nothing more to return - signal the iteration code to stop.
                    raise StopIteration()
            next_cut = self.cuts[next_cut_id]
            next_num_frames = num_frames + next_cut.num_frames
            next_num_cuts = len(cut_ids) + 1
            # Did we exceed the max_frames and max_cuts constraints?
            if next_num_frames <= self.max_frames and (self.max_cuts is None or next_num_cuts <= self.max_cuts):
                # No - add the next cut to the batch, and keep trying.
                num_frames = next_num_frames
                cut_ids.append(next_cut.id)
                self.current_idx += 1
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
                    self.current_idx += 1
        return cut_ids


class CutPairsSampler(CutSampler):
    """
    Samples pairs of cuts from a "source" and "target" CutSet.
    It expects that both CutSet's strictly consist of Cuts with corresponding IDs.
    It will try to satisfy the criteria of max_source_frames, max_target_frames, and max_cuts.
    It behaves like an iterable that yields lists of strings (cut IDs).
    """

    def __init__(
            self,
            source_cuts: CutSet,
            target_cuts: CutSet,
            max_source_frames: int = 26000,
            max_target_frames: int = 26000,
            max_cuts: Optional[int] = None,
            **kwargs
    ):
        """
        CutPairsSampler's constructor.

        :param source_cuts: the first ``CutSet`` to sample data from.
        :param target_cuts: the second ``CutSet`` to sample data from.
        :param max_source_frames: The maximum number of feature frames from ``source_cuts``
            that we're going to put in a single batch.
            The padding introduced during collation does not contribute to that limit.
        :param max_source_frames: The maximum number of feature frames from ``target_cuts``
            that we're going to put in a single batch.
            The padding introduced during collation does not contribute to that limit.
        :param max_cuts: The maximum number of cuts sampled to form a mini-batch.
            By default, this constraint is off.
        """
        super().__init__(source_cuts.ids, **kwargs)
        self.source_cuts = source_cuts
        self.target_cuts = target_cuts
        assert set(self.source_cuts.ids) == set(self.target_cuts.ids), \
            "Expected source and target cuts to have the same set of IDs."
        # Constraints
        self.max_source_frames = max_source_frames
        self.max_target_frames = max_target_frames
        self.max_cuts = max_cuts

    def __next__(self) -> List[str]:
        # Keep iterating the underlying CutSets as long as we hit or exceed the constraints
        # provided by user (the max number of source_feats or max number of cuts).
        # Note: no actual data is loaded into memory yet because the manifests contain all the metadata
        # required to do this operation.
        num_source_frames = 0
        num_target_frames = 0
        cut_ids = []
        while True:
            # Check that we have not reached the end of the dataset.
            if self.current_idx < len(self.data_source):
                # We didn't - grab the next cut
                next_cut_id = self.data_source[self.current_idx]
            else:
                if cut_ids:
                    # We did and we have a partial batch - return it.
                    return cut_ids
                else:
                    # We did and there is nothing more to return - signal the iteration code to stop.
                    raise StopIteration()
            next_source_cut = self.source_cuts[next_cut_id]
            next_target_cut = self.target_cuts[next_cut_id]
            next_num_source_frames = num_source_frames + next_source_cut.num_frames
            next_num_target_frames = num_target_frames + next_target_cut.num_frames
            next_num_cuts = len(cut_ids) + 1
            # Did we exceed the max_source_frames and max_cuts constraints?
            if next_num_source_frames <= self.max_source_frames \
                    and (next_num_target_frames <= self.max_target_frames) \
                    and (self.max_cuts is None or next_num_cuts <= self.max_cuts):
                # No - add the next cut to the batch, and keep trying.
                num_source_frames = next_num_source_frames
                num_target_frames = next_num_target_frames
                cut_ids.append(next_source_cut.id)
                self.current_idx += 1
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
                    self.current_idx += 1
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
            **kwargs
    ):
        """
        BucketingSampler's constructor.

        :param cuts: one or more ``CutSet`` objects.
            The first one will be used to determine the buckets for all of them.
            Then, all of them will be used to instantiate the per-bucket samplers.
        :param sampler_type: a sampler type that will be created for each underlying bucket.
        :param num_buckets: how many buckets to create.
        :param kwargs: Arguments used to create the underlying sampler for each bucket.
        """
        # Do not use the distributed capacities of the CutSampler in the top-level sampler.
        super().__init__(cuts[0].ids, world_size=1, rank=0)
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
            sampler_type(*bucket_cut_sets, **kwargs)
            for bucket_cut_sets in self.buckets
        ]
        self.depleted = [False] * num_buckets

    def set_epoch(self, epoch: int) -> None:
        for s in self.bucket_samplers:
            s.set_epoch(epoch)

    def __iter__(self) -> 'BucketingSampler':
        for b in self.bucket_samplers:
            iter(b)
        self.depleted = [False] * self.num_buckets
        return self

    def __next__(self) -> List[str]:
        while not self.is_depleted:
            idx, sampler = random.choice(self._nondepleted_samplers_with_idxs)
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
            (idx, bs) for idx, (bs, depleted) in
            enumerate(zip(self.bucket_samplers, self.depleted))
            if not depleted
        ]


def partition_cut_ids(
        data_source: List[str],
        world_size: int = 1,
        rank: int = 0
) -> List[str]:
    """
    Returns a list of cut IDs to be used by a single dataloading process.
    For multiple dataloader workers or ``DistributedDataParallel`` training,
    that list will be a subset of ``sampler.full_data_source``.

    :param data_source: a list of Cut IDs, representing the full dataset.
    :param world_size: Total number of distributed nodes. Set only when using ``DistributedDataParallel``.
    :param rank: Index of distributed node. Set only when using ``DistributedDataParallel``.
    """

    # First, split depending on the world_size and rank.
    if world_size == 1:
        return data_source
    else:
        # Distributed training is active - split full dataset into a subset.
        total = len(data_source)
        per_partition = int(ceil(total / float(world_size)))
        partition_start = rank * per_partition
        partition_end = min(partition_start + per_partition, total)
        logging.info(f'Distributed training with world size of {world_size} detected '
                     f'(node\'s local rank is {rank}. '
                     f'Splitting cuts into {world_size} partitions ('
                     f'this partition has cut IDs range [{partition_start, partition_end}].')

    return data_source[partition_start: partition_end]
