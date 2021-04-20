from math import ceil

import pytest
import numpy as np

from lhotse.utils import is_module_available
from lhotse.features.librosa_fbank import pad_or_truncate_features, LibrosaFbank


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


@pytest.mark.skipif(not is_module_available("librosa"), reason='Librosa is an optional dependency.')
@pytest.mark.parametrize(
    "audio_len",
    [22050, 11025, 1024, 512, 24000, 16000]
)
def test_librosa_fbank_with_different_audio_lengths(audio_len):

    extractor = LibrosaFbank()

    kernel_size = extractor.config.fft_size
    stride = extractor.config.hop_size
    pad = stride
    expected_n_frames = ceil((audio_len - kernel_size+ 2*pad) / stride + 1)

    n_frames = len(extractor.extract(np.zeros(audio_len), 22050))
    assert abs(n_frames - expected_n_frames) <= 1
