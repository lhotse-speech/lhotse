import pytest
import torch.utils.data

from lhotse import CutSet
from lhotse.dataset import IterableDatasetWrapper, SimpleCutSampler
from lhotse.testing.dummies import DummyManifest


class IdentityDataset(torch.utils.data.Dataset):
    def __getitem__(self, item):
        return item


@pytest.mark.parametrize("persistent_workers", [False, True])
def test_iterable_dataset_wrapper(persistent_workers):
    cuts = DummyManifest(CutSet, begin_id=0, end_id=10)
    sampler = SimpleCutSampler(cuts, max_cuts=10, shuffle=True)  # one batch
    dataset = IdentityDataset()
    dloader = torch.utils.data.DataLoader(
        IterableDatasetWrapper(
            dataset, sampler, auto_increment_epoch=persistent_workers
        ),
        batch_size=None,
        num_workers=1,
        persistent_workers=persistent_workers,
    )

    batches_per_epoch = []
    for epoch in range(2):
        dloader.dataset.set_epoch(epoch)
        batches = list(dloader)
        epoch_cuts = CutSet.from_cuts(c for b in batches for c in b)
        batches_per_epoch.append(epoch_cuts)

    assert list(batches_per_epoch[0].ids) != list(batches_per_epoch[1].ids)
