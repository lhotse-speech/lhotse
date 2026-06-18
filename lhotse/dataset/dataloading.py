import os
import random
import secrets
import sys
from functools import partial
from typing import Callable, Generator, Literal, Optional, Union

import torch
from torch import distributed as dist

from lhotse.utils import fix_random_seed

LHOTSE_PROCESS_SEED = "LHOTSE_PROCESS_SEED"

# Set by :func:`worker_init_fn` (called either by PyTorch's DataLoader in worker
# subprocesses or eagerly in the main process for the ``num_workers=0`` iterable
# path). Acts as the signal that :func:`get_worker_partition` should return a
# non-trivial ``(shard_id, num_shards)`` partition, so that indexed lazy iterators
# can split sample indices across DP rank x DataLoader worker. Map-style mode
# never calls ``worker_init_fn``, so this stays unset and partition collapses to
# ``(0, 1)`` — the sampler's own over-sample-and-discard handles DP dedup there.
LHOTSE_USE_WORKER_PARTITION = "LHOTSE_USE_WORKER_PARTITION"


def make_worker_init_fn(
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    set_different_node_and_worker_seeds: bool = True,
    seed: Optional[int] = 42,
) -> Optional[Callable[[int], None]]:
    """
    Calling this function creates a worker_init_fn suitable to pass to PyTorch's DataLoader.

    It helps with two issues:

    * sets the random seeds differently for each worker and node, which helps with
        avoiding duplication in randomized data augmentation techniques.
    * sets environment variables that help WebDataset detect it's inside multi-GPU (DDP)
        training, so that it correctly de-duplicates the data across nodes.
    """
    return partial(
        worker_init_fn,
        rank=rank,
        world_size=world_size,
        set_different_node_and_worker_seeds=set_different_node_and_worker_seeds,
        seed=seed,
    )


def worker_init_fn(
    worker_id: int,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    set_different_node_and_worker_seeds: bool = True,
    seed: Optional[int] = 42,
) -> None:
    """
    Function created by :func:`~lhotse.dataset.dataloading.make_worker_init_fn`, refer to its documentation for details.
    """
    if set_different_node_and_worker_seeds:
        process_seed = seed + 100 * worker_id
        if rank is not None:
            process_seed += 100000 * rank
        fix_random_seed(process_seed)
        os.environ[LHOTSE_PROCESS_SEED] = str(process_seed)

    if rank is None and world_size is None:
        return

    assert (
        rank is not None and world_size is not None
    ), f"Both args must be not None: rank={rank}, world_size={world_size}"

    # This sets the rank/world_size info for WebDataset to read it in worker subprocesses.
    # If we didn't do it, WebDataset will "think" this is always single-node training,
    # because DataLoader workers did not initialize torch.distributed.
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    # Signal that worker-level partition is active for indexed lazy iterators
    # (consumed by get_worker_partition). Map-style mode never calls this function,
    # so the flag stays unset and partition is (0, 1) there.
    os.environ[LHOTSE_USE_WORKER_PARTITION] = "1"


def resolve_seed(seed: Union[int, Literal["trng", "randomized"], None]) -> int:
    """
    Resolves the special values of random seed supported in Lhotse.

    If it's an integer, we'll just return it.

    If it's "trng", we'll use the ``secrets`` module to generate a random seed
    using a true RNG (to the extend supported by the OS).

    If it's "randomized", we'll check whether we're in a dataloading worker of ``torch.utils.data.DataLoader``.
    If we are, we expect that it was passed the result of :func:`~lhotse.dataset.dataloading.make_worker_init_fn`
    into its ``worker_init_fn`` argument, in which case we'll return a special seed exclusive to that worker.
    If we are not in a dataloading worker (or ``num_workers`` was set to ``0``), we'll return Python's ``random``
    module global seed.
    """

    # Specific number provided: use it.
    if isinstance(seed, int):
        return seed

    # No request for a specific type of random seed resolution: return Python's global random seed.
    if seed is None:
        return random.getstate()[1][0]

    # Deterministic randomized random seed resolution:
    # Each dataloading worker and DDP rank gets a separate random seed.
    # If we're not in a dataloading worker, use global RNG's current seed.
    if seed == "randomized":
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Not in a dataloader sub-process: get Python's global random seed.
            return random.getstate()[1][0]
        else:
            # In a dataloader sub-process: read out the seed we assigned to it.
            assert LHOTSE_PROCESS_SEED in os.environ, (
                "Requested seed='randomized' for shuffling shards differently "
                "on each DataLoader node and worker, "
                "but lhotse.dataset.dataloading.worker_init_fn was not called."
            )
            return int(os.environ[LHOTSE_PROCESS_SEED])

    # True-random number generator requested for seed generation ("complete randomness").
    if seed == "trng":
        # 2**32 may trigger the following exception if you add anything:
        # File "_mt19937.pyx", line 180, in numpy.random._mt19937.MT19937._legacy_seeding
        # ValueError: Seed must be between 0 and 2**32 - 1
        return secrets.randbelow(2**31)

    raise ValueError(
        f"Unexpected type or value of seed: {type(seed)=} {seed=}. "
        f"Supported values are: None, int, 'trng', and 'randomized'."
    )


def get_worker_partition() -> tuple:
    """
    Resolve the global ``(shard_id, num_shards)`` partition for the calling
    code, combining the DP rank with the DataLoader worker id.

    Returns ``(shard_id, num_shards)`` where
    ``shard_id = rank * num_workers + worker_id`` and
    ``num_shards = world_size * max(num_workers, 1)``.

    Returns the trivial ``(0, 1)`` partition when the ``LHOTSE_USE_WORKER_PARTITION``
    env var is not set — i.e. when :func:`worker_init_fn` has not been called.
    This keeps map-style mode (where the sampler runs in the main process and uses
    its own over-sample-and-discard DP dedup) unaffected even when RANK/WORLD_SIZE
    are already set in the environment (e.g. by torchrun).

    Used by indexed-manifest iterators (via :class:`~lhotse.indexing.LazyShuffledRange`)
    to deterministically split index ranges across DP ranks × DataLoader workers
    so each tuple yields a disjoint, non-overlapping subset.

    Reads DP info via :func:`get_rank` / :func:`get_world_size` (env-var aware;
    populated by :func:`worker_init_fn` inside DataLoader worker subprocesses).
    Reads the DataLoader worker info via :func:`torch.utils.data.get_worker_info`;
    when called outside a DataLoader worker (e.g. ``num_workers=0``), treats
    the caller as a single worker (``worker_id=0, num_workers=1``).
    """
    if os.environ.get(LHOTSE_USE_WORKER_PARTITION) != "1":
        return 0, 1
    rank = get_rank()
    world_size = get_world_size()
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        worker_id, num_workers = 0, 1
    else:
        worker_id = worker_info.id
        num_workers = max(worker_info.num_workers, 1)
    shard_id = rank * num_workers + worker_id
    num_shards = world_size * num_workers
    return shard_id, num_shards


class PartitionedIndexedIterator:
    """Shared partition-aware iteration driver for indexed leaf iterators.

    Encapsulates the (shard_id, num_shards) partition lookup, position
    tracking across DataLoader worker subprocesses, and topology-validated
    resume — the bits every indexed ``IteratorNode`` needs to repeat
    correctly. Yields global indices into the leaf source; the caller is
    responsible for decoding each index into the user-facing item.

    Two iteration modes are supported and selected at construction time:

    * **Stride** (``shuffle=False``, default): yields
      ``[shard_id, shard_id + num_shards, shard_id + 2 * num_shards, …]``
      — the simplest disjoint-per-rank partition.
    * **Feistel-shuffled** (``shuffle=True``, with ``seed``): yields a
      Feistel permutation of the full range restricted to this rank's
      slice, via :class:`~lhotse.indexing.LazyShuffledRange`. Useful when
      the underlying source is in a deterministic on-disk order but the
      consumer wants item-level shuffling within each shard.

    Typical wiring::

        class MyIndexedIterator(IteratorNode):
            def __init__(self, ...):
                ...
                self._iter_state = PartitionedIndexedIterator()

            def __iter__(self):
                for global_idx in self._iter_state.iterate(self._total_len):
                    item = self._decode_at(global_idx)
                    if item is None:
                        continue
                    yield item

            def state_dict(self) -> dict:
                return {**self._iter_state.state_dict(), "epoch": self.epoch}

            def load_state_dict(self, sd: dict) -> None:
                self._iter_state.load_state_dict(sd)
                self.epoch = sd.get("epoch", 0)

    Notes:
        * The partition is ``shard_id = rank * num_workers + worker_id``,
          ``num_shards = world_size * num_workers``. Outside DataLoader
          workers (or when :data:`LHOTSE_USE_WORKER_PARTITION` is unset —
          i.e. map-style mode) the partition collapses to ``(0, 1)`` and
          iteration covers the full range, matching pre-partition behavior.
        * ``state_dict`` stores the local-within-shard ``position`` plus
          the ``(shard_id, num_shards)`` topology captured at save time;
          on resume we refuse to continue under a different topology
          because the per-shard index sequence would diverge.
    """

    def __init__(self, shuffle: bool = False, seed: int = 0) -> None:
        self._shuffle = shuffle
        self._seed = seed
        self._position = 0
        self._shard_id: Optional[int] = None
        self._num_shards: Optional[int] = None
        self._restored = False
        # Constructed lazily inside :meth:`iterate` so the partition info
        # is read in the same process that owns the DataLoader worker env.
        self._range = None
        self._pending_range_state = None

    @property
    def position(self) -> int:
        """Local position within the current shard (0-indexed next element)."""
        return self._position

    def iterate(self, total_len: int) -> Generator[int, None, None]:
        """Yield global indices for this rank's slice of ``range(total_len)``.

        Raises:
            ValueError: if resuming from a saved state under a different
                ``(shard_id, num_shards)`` topology than the one recorded
                at save time.
        """
        shard_id, num_shards = get_worker_partition()

        if self._restored:
            self._restored = False
            if self._num_shards is not None and (
                self._shard_id != shard_id or self._num_shards != num_shards
            ):
                raise ValueError(
                    f"PartitionedIndexedIterator topology mismatch on resume: "
                    f"saved (shard_id={self._shard_id}, num_shards={self._num_shards}), "
                    f"current (shard_id={shard_id}, num_shards={num_shards}). "
                    f"Resuming with a different DP rank / DataLoader worker count "
                    f"is not supported (per-shard index sequence would diverge)."
                )
            start = self._position
        else:
            start = 0
            self._position = 0

        self._shard_id = shard_id
        self._num_shards = num_shards

        if self._shuffle:
            from lhotse.indexing import LazyShuffledRange

            self._range = LazyShuffledRange(
                total_len, seed=self._seed, shard_id=shard_id, num_shards=num_shards
            )
            if self._pending_range_state is not None:
                self._range.load_state_dict(self._pending_range_state)
                self._pending_range_state = None
            shard_len = len(self._range)
        else:
            self._range = None
            if total_len > shard_id:
                shard_len = (total_len - shard_id + num_shards - 1) // num_shards
            else:
                shard_len = 0

        for i in range(start, shard_len):
            self._position = i + 1
            if self._range is not None:
                yield self._range[i]
            else:
                yield shard_id + i * num_shards

    def state_dict(self) -> dict:
        sd = {
            "position": self._position,
            "shard_id": self._shard_id,
            "num_shards": self._num_shards,
        }
        if self._range is not None:
            sd["range"] = self._range.state_dict()
        elif self._pending_range_state is not None:
            sd["range"] = self._pending_range_state
        return sd

    def load_state_dict(self, sd: dict) -> None:
        self._position = sd.get("position", 0)
        self._shard_id = sd.get("shard_id")
        self._num_shards = sd.get("num_shards")
        if self._shuffle:
            # Stash the LazyShuffledRange state; it gets applied in iterate()
            # once the current (shard_id, num_shards) is known so the
            # topology-mismatch check there can fire first with a clearer
            # error than what LazyShuffledRange itself would raise.
            self._pending_range_state = sd.get("range")
            self._range = None
        self._restored = True


def get_world_size() -> int:
    """Source: https://github.com/danpovey/icefall/blob/74bf02bba6016c1eb37858a4e0e8a40f7d302bdb/icefall/dist.py#L56"""
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    else:
        return 1


def get_rank() -> int:
    """Source: https://github.com/danpovey/icefall/blob/74bf02bba6016c1eb37858a4e0e8a40f7d302bdb/icefall/dist.py#L56"""
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    elif dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    else:
        return 0
