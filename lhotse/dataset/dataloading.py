import os
import random
import secrets
import sys
from functools import partial
from typing import Callable, Literal, Optional, Union

import torch
from torch import distributed as dist

from lhotse.utils import fix_random_seed

LHOTSE_PROCESS_SEED = "LHOTSE_PROCESS_SEED"


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
