from math import ceil

import numpy as np
import pytest

from lhotse.features.whisper_fbank import WhisperFbank, WhisperFbankConfig
from lhotse.utils import is_module_available


@pytest.mark.skipif(
    not is_module_available("librosa"), reason="Librosa is an optional dependency."
)
@pytest.mark.parametrize("audio_len", [22050, 11025, 1024, 512, 24000, 16000])
def test_whisper_fbank_with_different_audio_lengths(audio_len):

    extractor = WhisperFbank(WhisperFbankConfig(device="cpu"))

    kernel_size = 400
    stride = extractor.hop_length
    pad = stride
    expected_n_frames = ceil((audio_len - kernel_size + 2 * pad) / stride + 1)

    n_frames = len(extractor.extract(np.zeros(audio_len, dtype=np.float32), 16000))
    assert abs(n_frames - expected_n_frames) <= 1
