import os
from functools import partial
from typing import Callable, Optional

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
    - sets the random seeds differently for each worker and node, which helps with
        avoiding duplication in randomized data augmentation techniques.
    - sets environment variables that help WebDataset detect it's inside multi-GPU (DDP)
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
