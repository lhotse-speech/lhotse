import os
from itertools import islice
from pathlib import Path
from typing import Optional, Union

from lhotse import Features, Recording
from lhotse.array import Array, TemporalArray
from lhotse.cut import Cut
from lhotse.utils import Pathlike


def fill_shar_placeholder(
    manifest: Union[Cut, Recording, Features, Array, TemporalArray],
    data: bytes,
    tarpath: Pathlike,
    field: Optional[str] = None,
) -> None:
    if isinstance(manifest, Cut):
        assert (
            field is not None
        ), "'field' argument must be provided when filling a Shar placeholder in a Cut."
        manifest = getattr(manifest, field)
        fill_shar_placeholder(
            manifest=manifest, field=field, data=data, tarpath=tarpath
        )

    tarpath = Path(tarpath)

    if isinstance(manifest, Recording):
        assert (
            len(manifest.sources) == 1
        ), "Multiple AudioSources are not supported yet."
        manifest.sources[0].type = "memory"
        manifest.sources[0].source = data

    elif isinstance(manifest, (Features, Array)):
        manifest.storage_key = data
        if tarpath.suffix == ".llc":
            manifest.storage_type = "memory_lilcom"
        elif tarpath.suffix == ".npy":
            manifest.storage_type = "memory_npy"
        else:
            raise RuntimeError(f"Unknown array/tensor format: {tarpath}")

    elif isinstance(manifest, TemporalArray):
        manifest.array.storage_key = data
        if tarpath.suffix == ".llc":
            manifest.array.storage_type = "memory_lilcom"
        elif tarpath.suffix == ".npy":
            manifest.array.storage_type = "memory_npy"
        else:
            raise RuntimeError(f"Unknown array/tensor format: {tarpath}")

    else:
        raise RuntimeError(f"Unknown manifest type: {type(manifest).__name__}")


def pytorch_worker_info(group=None):
    """
    Return node and worker info for PyTorch and some distributed environments.
    This function is copied from WebDataset: https://github.com/webdataset/webdataset
    """
    rank = 0
    world_size = 1
    worker = 0
    num_workers = 1
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        try:
            import torch.distributed

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                group = group or torch.distributed.group.WORLD
                rank = torch.distributed.get_rank(group=group)
                world_size = torch.distributed.get_world_size(group=group)
        except ModuleNotFoundError:
            pass
    if "WORKER" in os.environ and "NUM_WORKERS" in os.environ:
        worker = int(os.environ["WORKER"])
        num_workers = int(os.environ["NUM_WORKERS"])
    else:
        try:
            import torch.utils.data

            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                worker = worker_info.id
                num_workers = worker_info.num_workers
        except ModuleNotFoundError:
            pass

    return rank, world_size, worker, num_workers


def split_by_node(src, group=None):
    """
    This function is copied from WebDataset: https://github.com/webdataset/webdataset
    and adapted to lists.
    """
    rank, world_size, worker, num_workers = pytorch_worker_info(group=group)
    return src[rank::world_size]


def split_by_worker(src):
    """
    This function is copied from WebDataset: https://github.com/webdataset/webdataset
    and adapted to lists.
    """
    rank, world_size, worker, num_workers = pytorch_worker_info()
    return src[worker::num_workers]
