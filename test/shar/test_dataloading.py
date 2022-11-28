from pathlib import Path

import pytest
from torch.utils.data import DataLoader, Dataset

from lhotse import CutSet
from lhotse.dataset import (
    DynamicCutSampler,
    IterableDatasetWrapper,
    make_worker_init_fn,
)


class Identity(Dataset):
    def __getitem__(self, item):
        return item


@pytest.mark.parametrize("num_workers", [0, 1, 2])
def test_shar_lazy_reader_splits_by_dataloader_worker(shar_dir: Path, num_workers):
    # Prepare system under test
    cuts = CutSet.from_shar(
        fields={
            "cuts": [
                shar_dir / "cuts.000000.jsonl.gz",
                shar_dir / "cuts.000001.jsonl.gz",
            ],
            "recording": [
                shar_dir / "recording.000000.tar",
                shar_dir / "recording.000001.tar",
            ],
        },
        split_for_dataloading=True,
    )
    sampler = DynamicCutSampler(cuts, max_cuts=5)
    dloader = DataLoader(
        IterableDatasetWrapper(Identity(), sampler),
        batch_size=None,
        num_workers=num_workers,
        worker_init_fn=make_worker_init_fn(),
    )
    batches = []
    for batch in dloader:
        batches.append(batch)

    assert len(batches) == 4
    expected_cut_ids = sorted(cuts.ids)
    batch_cut_ids = sorted(c.id for b in batches for c in b)
    assert batch_cut_ids == expected_cut_ids


@pytest.mark.parametrize("randomize_seed", [True, False])
def test_shar_lazy_reader_different_shards_on_each_worker_with_randomized_seed(
    shar_dir: Path,
    randomize_seed: bool,
):
    num_workers = 2
    # seed value for which we found that both dataloader workers
    # shuffle the two shards differently
    base_seed = 0

    # Prepare system under test
    cuts = CutSet.from_shar(
        fields={
            "cuts": [
                shar_dir / "cuts.000000.jsonl.gz",
                shar_dir / "cuts.000001.jsonl.gz",
            ],
            "recording": [
                shar_dir / "recording.000000.tar",
                shar_dir / "recording.000001.tar",
            ],
        },
        seed="randomized" if randomize_seed else base_seed,
        shuffle_shards=True,
    )
    sampler = DynamicCutSampler(cuts, max_cuts=5)
    dloader = DataLoader(
        IterableDatasetWrapper(Identity(), sampler),
        batch_size=None,
        num_workers=num_workers,
        worker_init_fn=make_worker_init_fn(seed=base_seed),
    )
    batches = []
    for batch in dloader:
        batches.append(batch)

    # The batches contain data duplicated by the number of workers ...
    assert len(batches) == 8
    expected_cut_ids = sorted(cuts.ids)
    batch_cut_ids = sorted(c.id for b in batches for c in b)
    assert len(batch_cut_ids) == len(expected_cut_ids) * num_workers
    assert batch_cut_ids[::2] == expected_cut_ids
    assert batch_cut_ids[1::2] == expected_cut_ids

    # ... but that data was drawn from shards in a different order in each worker.
    batches_worker_0 = batches[::2]
    batches_worker_1 = batches[1::2]
    for b0, b1 in zip(batches_worker_0, batches_worker_1):
        for c0, c1 in zip(b0, b1):
            if randomize_seed:
                assert c0.shard_origin != c1.shard_origin
            else:
                assert c0.shard_origin == c1.shard_origin


def test_shar_cannot_randomize_seed_and_split_for_dataloading(shar_dir: Path):
    """
    Turning both options on would have resulted in a catastrophic data loss + duplication.
    """
    with pytest.raises(AssertionError):
        cuts = CutSet.from_shar(
            fields={
                "cuts": [
                    shar_dir / "cuts.000000.jsonl.gz",
                    shar_dir / "cuts.000001.jsonl.gz",
                ],
                "recording": [
                    shar_dir / "recording.000000.tar",
                    shar_dir / "recording.000001.tar",
                ],
            },
            seed="randomized",
            split_for_dataloading=True,
            shuffle_shards=True,
        )
