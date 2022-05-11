import os
import traceback
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional, Tuple

import pytest
import torch.distributed
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from lhotse import CutSet
from lhotse.dataset import SimpleCutSampler
from lhotse.dataset.webdataset import export_to_webdataset
from lhotse.utils import Pathlike


@pytest.mark.parametrize("world_size", [0, 1, 2])
@pytest.mark.parametrize("num_workers", [0, 1, 2])
def test_webdataset_deduplicates_data_in_ddp(world_size, num_workers):
    main(world_size=world_size, num_workers=num_workers)


def main(world_size: int, num_workers: int) -> None:
    """Sets up the data and spawns DDP processes."""
    print(f"Running test for: world_size={world_size}, num_workers={num_workers}")
    TOTAL_CUTS = 20
    with TemporaryDirectory() as root:
        n_shards, expected_cut_ids = prepare_data(total_cuts=TOTAL_CUTS, root=root)
        if world_size == 0:
            run_test(0, n_shards, root, None, expected_cut_ids, num_workers)
        else:
            mp.spawn(
                run_test,
                args=(
                    n_shards,
                    root,
                    world_size,
                    expected_cut_ids,
                    num_workers,
                ),
                nprocs=world_size,
                join=True,
            )


def prepare_data(total_cuts: int, root: Pathlike) -> Tuple[int, List[str]]:
    """
    Loads a cutset with 1 cut, repeats it a few times, and stores shards
    in tmp dir with 1 cut per shard for easy testing arithmetic.
    """
    cuts = CutSet.from_file("test/fixtures/libri/cuts_no_feats.json").repeat(total_cuts)
    Path(root).mkdir(exist_ok=True)
    n_shards = export_to_webdataset(
        cuts, f"{root}/shard-%06d.tar", shard_size=1, audio_format="wav", verbose=False
    )
    return n_shards, sorted(cuts.ids)


def run_test(
    rank: Optional[int],
    n_shards: int,
    root: str,
    world_size: Optional[int],
    expected_cut_ids: List[str],
    num_workers: int,
) -> None:

    # Initialize DDP if needed
    if world_size is not None:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12354"
        torch.distributed.init_process_group(
            "gloo",
            world_size=world_size,
            rank=rank,
        )
        # adjust the expected cut IDs according to rank
        expected_cut_ids_orig = expected_cut_ids
        expected_cut_ids = expected_cut_ids[rank::world_size]

    # Open CutSet with options that de-duplicate the data across nodes and workers
    cuts_wds = CutSet.from_webdataset(
        "%s/shard-{000000..%06d}.tar" % (root, n_shards - 1),
        split_by_node=True,
        split_by_worker=True,
    )

    # Iterate the data
    tot = 0
    cut_ids = []
    sampler = SimpleCutSampler(cuts_wds, max_duration=100, rank=0, world_size=1)
    dloader = DataLoader(
        DummyDataset(), sampler=sampler, batch_size=None, num_workers=num_workers
    )
    for batch in dloader:
        tot += len(batch)
        for c in batch:
            cut_ids.append(c.id)

    print(f"[Rank {rank}/{world_size}] Actual   cuts: ", sorted(cut_ids))
    print(f"[Rank {rank}/{world_size}] Expected cuts: ", sorted(expected_cut_ids))
    try:
        assert tot == len(expected_cut_ids)
        assert sorted(cut_ids) == sorted(
            expected_cut_ids
        ), f"{sorted(cut_ids)}\n!=\n{sorted(expected_cut_ids)}"
    except AssertionError:
        # Pytest doesn't work great with subprocesses
        print(traceback.print_exc())
        raise


class DummyDataset:
    """Dataset that returns input cuts."""

    def __getitem__(self, item):
        return item
