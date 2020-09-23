import numpy as np
import pytest

from lhotse.augmentation import WavAugmenter, is_wav_augment_available
from lhotse.cut import CutSet
from lhotse.dataset import UnsupervisedDataset, UnsupervisedWaveformDataset
from lhotse.dataset.unsupervised import DynamicUnsupervisedDataset
from lhotse.features import Fbank


@pytest.fixture
def libri_cut_set():
    return CutSet.from_json('test/fixtures/libri/cuts.json')


def test_unsupervised_dataset(libri_cut_set):
    dataset = UnsupervisedDataset(libri_cut_set)
    assert len(dataset) == 1
    feats = dataset[0]
    assert feats.shape == (1000, 40)


def test_unsupervised_waveform_dataset(libri_cut_set):
    dataset = UnsupervisedWaveformDataset(libri_cut_set)
    assert len(dataset) == 1
    audio = dataset[0]
    assert audio.shape == (1, 10 * 16000)


def test_on_the_fly_feature_extraction_unsupervised_dataset(libri_cut_set):
    ref_dataset = UnsupervisedDataset(libri_cut_set)
    tested_dataset = DynamicUnsupervisedDataset(
        feature_extractor=Fbank(),
        cuts=libri_cut_set
    )
    ref_feats = ref_dataset[0]
    tested_feats = tested_dataset[0]
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


@pytest.mark.skipif(not is_wav_augment_available(), reason='Requires WavAugment')
def test_on_the_fly_feature_extraction_unsupervised_dataset_with_augmentation(libri_cut_set):
    tested_dataset = DynamicUnsupervisedDataset(
        feature_extractor=Fbank(),
        cuts=libri_cut_set,
        augmenter=WavAugmenter.create_predefined('reverb', sampling_rate=16000)
    )
    # Just test that it runs
    tested_feats = tested_dataset[0]
