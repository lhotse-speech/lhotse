from pathlib import Path

import pytest
from torch.utils.data import DataLoader, Dataset

from lhotse import CutSet
from lhotse.dataset import (
    DynamicCutSampler,
    IterableDatasetWrapper,
    make_worker_init_fn,
)
from lhotse.shar.readers.lazy import LazySharIterator


class Identity(Dataset):
    def __getitem__(self, item):
        return item


@pytest.mark.parametrize("num_workers", [0, 1, 2])
def test_shar_lazy_reader_splits_by_dataloader_worker(shar_dir: Path, num_workers):
    # Prepare system under test
    cuts = CutSet(
        LazySharIterator(
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
