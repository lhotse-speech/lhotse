import os


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
