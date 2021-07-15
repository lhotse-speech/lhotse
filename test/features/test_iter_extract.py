import numpy as np
import pytest

from lhotse import Fbank, KaldiFbank, KaldiFbankConfig


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
    assert diff < 1e-10


@pytest.mark.parametrize('fbank', [Fbank(), KaldiFbank(KaldiFbankConfig(sampling_rate=8000))])
@pytest.mark.parametrize('num_samples', [7500, 7900, 7995, 8000, 8005, 8100, 8500, 15950, 16000, 16020, 16050, 16091])
def test_iter_extract_edges(fbank, num_samples):
    sampling_rate = 8000
    audio = np.random.randn(1, num_samples)
    feats = fbank.iter_extract(audio, sampling_rate=sampling_rate)
    feats_offline = fbank.extract(audio, sampling_rate=sampling_rate)
    diff = np.mean(np.abs(feats - feats_offline))
    assert diff < 2e-3
