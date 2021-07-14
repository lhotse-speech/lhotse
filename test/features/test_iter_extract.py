import numpy as np
import pytest

from lhotse import Fbank


@pytest.mark.parametrize(
    "sampling_rate",
    [
        8000,
        pytest.param(11025, marks=pytest.mark.xfail),
        16000,
        pytest.param(22050, marks=pytest.mark.xfail),
        24000,
        pytest.param(44100, marks=pytest.mark.xfail),
        48000,
    ],
)
def test_iter_extract_fbank(sampling_rate):
    duration = 800
    fbank = Fbank()
    audio = np.random.randn(1, sampling_rate * duration)
    feats = fbank.iter_extract(audio, sampling_rate=sampling_rate)
    feats_offline = fbank.extract(audio, sampling_rate=sampling_rate)
    diff = np.mean(np.abs(feats - feats_offline))
    print(diff)
    assert diff < 1e-10
