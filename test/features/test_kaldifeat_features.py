from tempfile import NamedTemporaryFile

import numpy as np
import pytest
import torch

from lhotse import Fbank, KaldifeatFbank, KaldifeatFbankConfig, Mfcc
from lhotse.features import create_default_feature_extractor
from lhotse.features.kaldifeat import KaldifeatMelOptions, KaldifeatMfcc
from lhotse.utils import nullcontext as does_not_raise

kaldifeat = pytest.importorskip(
    "kaldifeat", reason="Kaldifeat tests require kaldifeat to be installed."
)


@pytest.mark.parametrize(
    ["feature_type", "exception_expectation"],
    [
        ("kaldifeat-fbank", does_not_raise()),
        ("kaldifeat-mfcc", does_not_raise()),
    ],
)
def test_feature_extractor(feature_type, exception_expectation):
    import soundfile as sf

    # For now, just test that it runs
    with exception_expectation:
        fe = create_default_feature_extractor(feature_type)
        samples, sr = sf.read("test/fixtures/libri/libri-1088-134315-0000.wav")
        fe.extract(samples=samples, sampling_rate=sr)


def test_kaldifeat_config():
    x = np.arange(8000, dtype=np.float32)
    fe = KaldifeatFbank(KaldifeatFbankConfig(mel_opts=KaldifeatMelOptions(num_bins=27)))
    feats = fe.extract(x, sampling_rate=16000)
    assert feats.shape == (50, 27)


@pytest.mark.parametrize(
    "input",
    [
        np.arange(8000, dtype=np.float32),
        torch.arange(8000, dtype=torch.float32),
        torch.arange(8000, dtype=torch.float32),
    ],
)
def test_kaldifeat_supports_single_input_waveform(input):
    fe = KaldifeatFbank()
    feats = fe.extract(input, sampling_rate=16000)
    assert feats.shape == (50, 80)


@pytest.mark.parametrize(
    "input",
    [
        [np.arange(8000, dtype=np.float32)],
        [np.arange(8000, dtype=np.float32).reshape(1, -1)],
        [torch.arange(8000, dtype=torch.float32).unsqueeze(0)],
    ],
)
def test_kaldifeat_supports_list_with_single_input_waveform(input):
    fe = KaldifeatFbank()
    feats = fe.extract(input, sampling_rate=16000)
    assert isinstance(feats, list)
    assert len(feats) == 1
    assert feats[0].shape == (50, 80)


@pytest.mark.parametrize(
    "input",
    [
        [
            np.arange(8000, dtype=np.float32),
            np.arange(8000, dtype=np.float32),
        ],
        [
            torch.arange(8000, dtype=torch.float32),
            torch.arange(8000, dtype=torch.float32),
        ],
    ],
)
def test_kaldifeat_supports_list_of_even_len_inputs(input):
    fe = KaldifeatFbank()
    feats = fe.extract(input, sampling_rate=16000)
    assert feats.ndim == 3
    assert feats.shape == (2, 50, 80)


def test_kaldifeat_supports_list_of_uneven_len_inputs():
    input = [
        torch.arange(8000, dtype=torch.float32),
        torch.arange(16000, dtype=torch.float32),
    ]
    fe = KaldifeatFbank()
    feats = fe.extract(input, sampling_rate=16000)
    assert len(feats) == 2
    f1, f2 = feats
    assert f1.shape == (50, 80)
    assert f2.shape == (100, 80)


@pytest.mark.parametrize(
    ["extractor1", "extractor2"],
    [
        (KaldifeatFbank(), Fbank()),
        (KaldifeatMfcc(), Mfcc()),
    ],
)
def test_kaldifeat_torchaudio_equivalence(extractor1, extractor2):
    sampling_rate = 16000
    np.random.seed(99)  # ensure reproducibility
    audio = np.random.rand(1, 32000).astype(np.float32)
    feat1 = extractor1.extract(audio, sampling_rate)
    feat2 = extractor2.extract(audio, sampling_rate)
    np.testing.assert_almost_equal(feat1, feat2, decimal=3)


@pytest.mark.parametrize("feature_type", ["kaldifeat-fbank", "kaldifeat-mfcc"])
def test_feature_extractor_serialization(feature_type):
    fe = create_default_feature_extractor(feature_type)
    with NamedTemporaryFile() as f:
        fe.to_yaml(f.name)
        fe_deserialized = type(fe).from_yaml(f.name)
    assert fe_deserialized.config == fe.config
