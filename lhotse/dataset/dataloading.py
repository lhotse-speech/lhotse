import platform
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from typing import Any, Dict, List

import torch.utils.data

from lhotse.dataset.sampling.base import CutSampler


class LhotseDataLoader:
    """
    A simplified ``DataLoader`` implementation that relies on a ``ProcessPoolExecutor``.
    The main difference between this and ``torch.utils.data.DataLoader`` is that
    :class:`.LhotseDataLoader` allows to launch subprocesses inside of its workers.
    This is useful for working with dataset classes which perform dynamic batching
    and need to perform concurrent I/O to read all the necessary data from disk/network.

    .. note:: :class:`.LhotseDataLoader` does not support ``num_workers=0``.

    .. warning:: :class:`.LhotseDataLoader` is experimental and not guaranteed to work
        correctly across all possible edge cases related to subprocess worker termination.
        If you experience stability problems, contact us or use a standard ``DataLoader``
        instead.

    .. warning:: :class:`.LhotseDataLoader` requires Python >= 3.7.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        sampler: CutSampler,
        num_workers: int = 1,
        prefetch_factor: int = 2,
    ) -> None:
        from packaging.version import parse as _version

        if _version(platform.python_version()) < _version("3.7"):
            raise RuntimeError("LhotseDataLoader requires Python version at least 3.7")
        assert num_workers >= 1
        assert prefetch_factor >= 1
        self.dataset = dataset
        self.sampler = sampler
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        # Mutable state
        self._iter = None
        self._futures = deque([])
        # Start the worker processes. The initializer receives the dataset object
        # from the main process and caches it globally, so that it can be re-used
        # for subsequent tasks sent to the worker. This helps avoid excessive
        # communication between the processes.
        self.pool = ProcessPoolExecutor(
            num_workers,
            initializer=_init_worker,
            initargs=(dataset,),
            mp_context=get_context("spawn"),
        )

    def __iter__(self) -> "LhotseDataLoader":
        """Prepares the sampler for iteration and schedules initial tasks to the workers."""
        self._iter = iter(self.sampler)
        for _ in range(self.prefetch_factor * self.num_workers):
            self._schedule_one()
        return self

    def _schedule_one(self) -> None:
        """Submits a task and stores the future for results retrieval."""
        if self._iter is not None:
            try:
                self._futures.append(self.pool.submit(_get_item, next(self._iter)))
            except StopIteration:
                self._iter = None

    def _retrieve_one(self) -> Dict[str, Any]:
        """Retrieves the result from the earliest submitted task."""
        if self._futures:
            return self._futures.popleft().result()
        raise StopIteration()

    def __next__(self) -> Dict[str, Any]:
        """Submits a new batch to process and then retrieves and returns a completed batch."""
        self._schedule_one()
        return self._retrieve_one()


def _init_worker(dataset: torch.utils.data.Dataset) -> None:
    """
    Stores the dataset in the global state of the process -- this is safe because
    the process is initialized only once and used for unique dataset in its life span.
    """
    global _GLOBAL_DATASET_CACHE
    _GLOBAL_DATASET_CACHE = dataset


def _get_item(cut_ids: List[str]) -> Dict[str, Any]:
    """
    Queries the globally cached dataset to retrieve a batch. Has to be run
    inside a worker process that was initialized with :meth:`._init_worker`.
    """
    return _GLOBAL_DATASET_CACHE[cut_ids]


_GLOBAL_DATASET_CACHE = None
