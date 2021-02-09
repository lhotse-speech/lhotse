import logging
import math
import random
import warnings
from abc import ABC, abstractmethod
from typing import Any, Iterable, List, Optional, Tuple

import torch

from lhotse import CutSet


class CutSampler(ABC):
    """
    CutSampler is responsible for collecting batches of cuts, given specified criteria.
    It implements correct handling of multiprocessing in DataLoader,
    as well as distributed training, so that the cuts are not duplicated across workers.

    Sampling in a CutSampler is intended to be very quick - it only uses the metadata in
    ``CutSet`` manifest to select the cuts, and is not intended to perform any I/O.

    CutSampler works similarly to PyTorch's DistributedSampler - when :attr:`shuffle=True`,
    you should call ``sampler.set_epoch(epoch)`` at each new epoch to have a different
    ordering of returned elements.

    Example usage::

        >>> dataset = K2SpeechRecognitionIterableDataset(
        ...     cuts,
        ...     sampler=SingleCutSampler(cuts, world_size=4, local_rank=0)
        ... )
        >>> loader = DataLoader(dataset, batch_size=None)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     dataset.sampler.set_epoch(epoch)
        ...     train(loader)

    .. note::

        For implementers of new samplers:
        Subclasses of CutSampler are expected to implement ``__next__()`` to introduce specific
        sampling logic (e.g. based on filters such as max number of frames/tokens/etc.).
        CutSampler defines ``__iter__()`` which optionally shuffles the cut IDs, and resets
        ``self.current_idx`` to zero (to be used and incremented inside of ``__next__()``.
    """

    def __init__(
            self,
            cut_ids: Iterable[str],
            shuffle: bool = False,
            world_size: int = 1,
            local_rank: int = 0,
            seed: int = 0,
            epoch: int = 0
    ) -> None:
        """

        :param cut_ids: An iterable of cut IDs for the full dataset.
            CutSampler will take care of partitioning that into parallel/distributed workers.
        :param shuffle: When ``True``, the cuts will be shuffled at the start of iteration.
            Convenient when mini-batch loop is inside an outer epoch-level loop, e.g.:
            `for epoch in range(10): for batch in dataset: ...` as every epoch will see a
            different cuts order.
        :param world_size: Total number of distributed nodes. Set only when using ``DistributedDataParallel``.
        :param local_rank: Index of distributed node. Set only when using ``DistributedDataParallel``.
        :param seed: Random seed used to consistently shuffle the dataset across different processes.
        """
        self.all_cut_ids = list(cut_ids)
        self.shuffle = shuffle
        self.world_size = world_size
        self.local_rank = local_rank
        self.seed = seed
        self.epoch = epoch

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        :param epoch: Epoch number.
        """
        self.epoch = epoch

    def __iter__(self) -> 'CutSampler':
        """
        Prepare the dataset for iterating over a new epoch. Will shuffle the data if requested.
        """
        # We can only retrieve the cut ids once iter is called, not before!
        # Iter is called inside a dataloader process, whereas init is called when
        # instantiating this class (typically before dataloader).
        self.cut_ids = partition_cut_ids(self.all_cut_ids, world_size=self.world_size, local_rank=self.local_rank)
        if self.shuffle:
            r = random.Random(self.seed + self.epoch)
            r.shuffle(self.cut_ids)
        self.current_idx = 0
        return self

    @abstractmethod
    def __next__(self) -> Any:
        pass


class SingleCutSampler(CutSampler):
    """
    Samples cuts from a CutSet to satisfy the criteria of max_frames and max_cuts.
    Use as an iterable that yields ``CutSet`` objects (individual batches).
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
        :param max_frames: The maximum number of feature frames that we're going to put in a single batch.
            The padding frames do not contribute to that limit, since we pack the batch by default to minimze
            the amount of padding.
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

    def __next__(self) -> CutSet:
        # Keep iterating the underlying CutSet as long as we hit or exceed the constraints
        # provided by user (the max number of frames or max number of cuts).
        # Note: no actual data is loaded into memory yet because the manifests contain all the metadata
        # required to do this operation.
        num_frames = 0
        cuts = []
        while True:
            # Check that we have not reached the end of the dataset.
            if self.current_idx < len(self.cut_ids):
                # We didn't - grab the next cut
                next_cut_id = self.cut_ids[self.current_idx]
            else:
                if cuts:
                    # We did and we have a partial batch - return it.
                    return CutSet.from_cuts(cuts)
                else:
                    # We did and there is nothing more to return - signal the iteration code to stop.
                    raise StopIteration()
            next_cut = self.cuts[next_cut_id]
            next_num_frames = num_frames + next_cut.num_frames
            next_num_cuts = len(cuts) + 1
            # Did we exceed the max_frames and max_cuts constraints?
            if next_num_frames <= self.max_frames and (self.max_cuts is None or next_num_cuts <= self.max_cuts):
                # No - add the next cut to the batch, and keep trying.
                num_frames = next_num_frames
                cuts.append(next_cut)
                self.current_idx += 1
            else:
                # Yes. Do we have at least one cut in the batch?
                if cuts:
                    # Yes. Return it.
                    break
                else:
                    # No. We'll warn the user that the constrains might be too tight,
                    # and return the cut anyway.
                    warnings.warn("The first cut drawn in batch collection violates the max_frames or max_cuts "
                                  "constraints - we'll return it anyway. Consider increasing max_frames/max_cuts.")
                    cuts.append(next_cut)
                    self.current_idx += 1
        batch = CutSet.from_cuts(cuts)
        return batch


class CutPairsSampler(CutSampler):
    """
    Samples pairs of cuts from a "source" and "target" CutSet.
    It expects that both CutSet's strictly consist of Cuts with corresponding IDs.
    It will try to satisfy the criteria of max_source_frames, max_target_frames, and max_cuts.
    Use as an iterable that yields pairs of ``CutSet`` objects (individual batches).
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
        :param max_frames: The maximum number of feature frames that we're going to put in a single batch.
            The padding frames do not contribute to that limit, since we pack the batch by default to minimze
            the amount of padding.
        :param max_cuts: The maximum number of cuts sampled to form a mini-batch.
            By default, this constraint is off.
        :param shuffle: When ``True``, the cuts will be shuffled at the start of iteration.
            Convenient when mini-batch loop is inside an outer epoch-level loop, e.g.:
            `for epoch in range(10): for batch in dataset: ...` as every epoch will see a
            different cuts order.
        """
        super().__init__(self.source_cuts.ids, **kwargs)
        self.source_cuts = source_cuts
        self.target_cuts = target_cuts
        assert set(self.source_cuts.ids) == set(self.target_cuts.ids), \
            "Expected source and target cuts to have the same set of IDs."
        # Constraints
        self.max_source_frames = max_source_frames
        self.max_target_frames = max_target_frames
        self.max_cuts = max_cuts

    def __next__(self) -> Tuple[CutSet, CutSet]:
        # Keep iterating the underlying CutSet as long as we hit or exceed the constraints
        # provided by user (the max number of source_feats or max number of cuts).
        # Note: no actual data is loaded into memory yet because the manifests contain all the metadata
        # required to do this operation.
        num_source_frames = 0
        num_target_frames = 0
        source_cuts = []
        while True:
            # Check that we have not reached the end of the dataset.
            if self.current_idx < len(self.cut_ids):
                # We didn't - grab the next cut
                next_cut_id = self.cut_ids[self.current_idx]
            else:
                if source_cuts:
                    # We did and we have a partial batch - return it.
                    return (
                        CutSet.from_cuts(source_cuts),
                        CutSet.from_cuts(self.target_cuts[c.id] for c in source_cuts)
                    )
                else:
                    # We did and there is nothing more to return - signal the iteration code to stop.
                    raise StopIteration()
            next_source_cut = self.source_cuts[next_cut_id]
            next_target_cut = self.target_cuts[next_cut_id]
            next_num_source_frames = num_source_frames + next_source_cut.num_frames
            next_num_target_frames = num_target_frames + next_target_cut.num_frames
            next_num_cuts = len(source_cuts) + 1
            # Did we exceed the max_source_frames and max_cuts constraints?
            if next_num_source_frames <= self.max_source_frames \
                    and (next_num_target_frames <= self.max_target_frames) \
                    and (self.max_cuts is None or next_num_cuts <= self.max_cuts):
                # No - add the next cut to the batch, and keep trying.
                num_source_frames = next_num_source_frames
                num_target_frames = next_num_target_frames
                source_cuts.append(next_source_cut)
                self.current_idx += 1
            else:
                # Yes. Do we have at least one cut in the batch?
                if source_cuts:
                    # Yes. Return it.
                    break
                else:
                    # No. We'll warn the user that the constrains might be too tight,
                    # and return the cut anyway.
                    warnings.warn("The first cut drawn in batch collection violates one of the max_... constraints"
                                  "we'll return it anyway. Consider increasing max_source_frames/max_cuts/etc.")
                    source_cuts.append(next_source_cut)
                    self.current_idx += 1
        return (
            CutSet.from_cuts(source_cuts),
            CutSet.from_cuts(self.target_cuts[c.id] for c in source_cuts)
        )


def partition_cut_ids(
        cut_ids: List[str],
        world_size: int = 1,
        local_rank: int = 0
) -> List[str]:
    """
    Returns a list of cut IDs to be used by a single dataloading process.
    For multiple dataloader workers or ``DistributedDataParallel`` training,
    that list will be smaller than ``self.cut_ids``.

    This method takes care of partitioning for multiprocess data loading, so that the
    dataset won't return data duplicates within a single epoch (for more details, see:
    https://pytorch.org/docs/stable/data.html at "Multi-process data loading").

    :param cut_ids: a list of Cut IDs, representing the full dataset.
    :param world_size: Total number of distributed nodes. Set only when using ``DistributedDataParallel``.
    :param local_rank: Index of distributed node. Set only when using ``DistributedDataParallel``.
    """

    # First, split depending on the world_size and local_rank.
    if world_size == 1:
        logging.info('No distributed training - not partitioning here.')
        partition_start = 0
        partition_end = len(cut_ids)
    else:
        logging.info(f'Distributed training with world size of {world_size} detected '
                     f'(node\'s local rank is {local_rank}. '
                     f'Splitting cuts into {world_size} partitions.')
        # Distributed training is active - split full dataset into a subset.
        total = len(cut_ids)
        per_partition = int(math.ceil(total / float(world_size)))
        partition_start = local_rank * per_partition
        partition_end = min(partition_start + per_partition, total)

    # Ask PyTorch about multiprocessing in the DataLoader.
    # If there is no multiprocessing involved, we'll iterate full partition for this node.
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        logging.info(f'Multiprocessing dataloader detected with num workers of {worker_info.num_workers}. '
                     f'Creating per-worker partitions (the effect stacks with distributed training, if used).')
        # We are in a worker process - need to select a partition to process.
        per_worker = int(math.ceil((partition_end - partition_start) / float(worker_info.num_workers)))
        worker_id = worker_info.id
        partition_start += worker_id * per_worker
        partition_end = min(partition_start + per_worker, partition_end)

    return cut_ids[partition_start: partition_end]
