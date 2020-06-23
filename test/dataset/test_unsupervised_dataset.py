import pytest

from lhotse.cut import CutSet
from lhotse.dataset import UnsupervisedDataset


@pytest.fixture
def libri_cut_set():
    return CutSet.from_yaml('test/fixtures/libri/cuts.yml')


def test_unsupervised_dataset(libri_cut_set):
    dataset = UnsupervisedDataset(libri_cut_set)
    assert len(dataset) == 1
    feats = dataset[0]
    assert feats.shape == (1000, 23)
