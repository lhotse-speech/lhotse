import pytest

from lhotse.cut import CutSet
from lhotse.dataset import SingleCutSampler, UnsupervisedDataset
from lhotse.dataset.dataloading import LhotseDataLoader


@pytest.fixture
def cuts():
    cuts = CutSet.from_file("test/fixtures/libri/cuts.json")
    return sum(
        (cuts.modify_ids(lambda cid: cid + str(i)) for i in range(100)), start=CutSet()
    )


@pytest.fixture
def dataset(cuts):
    return UnsupervisedDataset(cuts)


def test_lhotse_dataloader_runs(cuts, dataset):
    sampler = SingleCutSampler(cuts, max_cuts=2)
    dloader = LhotseDataLoader(dataset, sampler, num_workers=2)
    batches = list(dloader)
    assert len(batches) == 50
