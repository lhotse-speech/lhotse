import pytest
import numpy as np

from lhotse.features.librosa_fbank import pad_or_truncate_features


@pytest.mark.parametrize(
    "feats,expected_num_frames,abs_tol",
    [
        (np.zeros((5, 2)), 5, 0),
        (np.zeros((5, 2)), 4, 1),
        (np.zeros((5, 2)), 6, 1),
        (np.zeros((5, 2)), 3, 2),
        (np.zeros((5, 2)), 7, 2),
    ]
)
def test_pad_or_truncate_features_shape(feats, expected_num_frames, abs_tol):
    feats_adjusted = pad_or_truncate_features(feats, expected_num_frames, abs_tol)
    assert feats_adjusted.shape == (expected_num_frames, feats.shape[-1])


@pytest.mark.parametrize(
    "feats,expected_num_frames,abs_tol",
    [
        (np.zeros((5, 2)), 4, 0),
        (np.zeros((5, 2)), 3, 1),
        (np.zeros((5, 2)), 7, 1),
        (np.zeros((5, 2)), 2, 2),
        (np.zeros((5, 2)), 8, 2),
    ]
)
def test_pad_or_truncate_features_fails(feats, expected_num_frames, abs_tol):
    with pytest.raises(ValueError):
        pad_or_truncate_features(feats, expected_num_frames, abs_tol)
