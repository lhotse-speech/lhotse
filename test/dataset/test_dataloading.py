import platform

import pytest
from packaging.version import parse as _version

from lhotse.cut import CutSet
from lhotse.dataset import SingleCutSampler, UnsupervisedDataset
from lhotse.dataset.dataloading import LhotseDataLoader


@pytest.fixture
def cuts():
    cuts = CutSet.from_file("test/fixtures/libri/cuts.json")
    # Concatenate 100 cut sets together, starting with an empty CutSet()
    return sum(
        (cuts.modify_ids(lambda cid: cid + str(i)) for i in range(100)), CutSet()
    )


@pytest.mark.skipif(
    _version(platform.python_version()) < _version("3.7"),
    reason="LhotseDataLoader requires Python 3.7+",
)
def test_lhotse_dataloader_runs(cuts):
    sampler = SingleCutSampler(cuts, max_cuts=2)
    dloader = LhotseDataLoader(UnsupervisedDataset(), sampler, num_workers=2)
    batches = list(dloader)
    assert len(batches) == 50
