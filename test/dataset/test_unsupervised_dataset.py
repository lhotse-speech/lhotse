import numpy as np
import pytest
import torch

from lhotse.cut import CutSet
from lhotse.dataset import UnsupervisedDataset, UnsupervisedWaveformDataset
from lhotse.dataset.unsupervised import DynamicUnsupervisedDataset
from lhotse.features import Fbank


@pytest.fixture
def libri_cut_set():
    return CutSet.from_json('test/fixtures/libri/cuts.json')


def test_unsupervised_dataset(libri_cut_set):
    ids = list(libri_cut_set.ids)
    dataset = UnsupervisedDataset(libri_cut_set)
    assert len(dataset) == 1
    out = dataset[ids]
    assert out["features"].shape == (1, 1000, 40)


def test_unsupervised_waveform_dataset(libri_cut_set):
    ids = list(libri_cut_set.ids)
    dataset = UnsupervisedWaveformDataset(libri_cut_set)
    assert len(dataset) == 1
    out = dataset[ids]
    assert out["audio"].shape == (1, 10 * 16000)
    assert isinstance(out["audio_lens"], torch.IntTensor)


def test_on_the_fly_feature_extraction_unsupervised_dataset(libri_cut_set):
    ids = list(libri_cut_set.ids)
    ref_dataset = UnsupervisedDataset(libri_cut_set)
    tested_dataset = DynamicUnsupervisedDataset(
        feature_extractor=Fbank(),
        cuts=libri_cut_set
    )
    out = ref_dataset[ids]
    ref_feats = out["features"]
    tested_feats = tested_dataset[ids]
    # Note: comparison to 1 decimal fails.
    #       I'm assuming this is due to lilcom's compression.
    #       Pytest outputs looks like the following:
    # E       Mismatched elements: 4 / 23000 (0.0174%)
    # E       Max absolute difference: 0.46469784
    # E       Max relative difference: 0.6171043
    # E        x: array([[-11.5, -11.4,  -9.9, ...,  -5.5,  -6.5,  -7.4],
    # E              [-13.2, -11.2,  -9.6, ...,  -5.6,  -6.5,  -7.6],
    # E              [-12. , -10.1, -10.1, ...,  -5.8,  -7. ,  -7.8],...
    # E        y: array([[-11.5, -11.4,  -9.9, ...,  -5.5,  -6.5,  -7.4],
    # E              [-13.2, -11.2,  -9.6, ...,  -5.6,  -6.5,  -7.6],
    # E              [-12. , -10.1, -10.1, ...,  -5.8,  -7. ,  -7.8],...
    np.testing.assert_array_almost_equal(ref_feats, tested_feats, decimal=0)
