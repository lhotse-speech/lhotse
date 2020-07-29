import pytest

from lhotse.cut import CutSet
from lhotse.dataset.unsupervised import UnsupervisedDataset, UnsupervisedWaveformDataset


@pytest.fixture
def libri_cut_set():
    return CutSet.from_yaml('test/fixtures/libri/cuts.yml')


def test_unsupervised_dataset(libri_cut_set):
    dataset = UnsupervisedDataset(libri_cut_set)
    assert len(dataset) == 1
    feats = dataset[0]
    assert feats.shape == (1000, 23)


def test_unsupervised_waveform_dataset(libri_cut_set):
    dataset = UnsupervisedWaveformDataset(libri_cut_set)
    assert len(dataset) == 1
    audio = dataset[0]
    assert audio.shape == (1, 10 * 16000)
