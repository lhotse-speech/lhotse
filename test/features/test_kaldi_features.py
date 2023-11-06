from tempfile import NamedTemporaryFile

import numpy as np
import pytest
import torch

from lhotse import Recording, TorchaudioFbank, TorchaudioMfcc, TorchaudioSpectrogram
from lhotse.features import create_default_feature_extractor
from lhotse.features.kaldi.extractors import (
    Fbank,
    FbankConfig,
    LogSpectrogram,
    LogSpectrogramConfig,
    Mfcc,
    MfccConfig,
    Spectrogram,
    SpectrogramConfig,
)
from lhotse.features.kaldi.layers import Wav2LogFilterBank, Wav2MFCC, Wav2Spec
from lhotse.testing.random import deterministic_rng


@pytest.fixture()
def recording():
    return Recording.from_file("test/fixtures/libri/libri-1088-134315-0000.wav")


def test_kaldi_fbank_layer(recording):
    # Prepare a batch of recordings
    audio = torch.from_numpy(recording.load_audio())
    audio = torch.cat([audio] * 4, dim=0)
    assert audio.shape == (4, recording.num_samples)
    # We'll test the Kaldi feature extraction layers
    # by checking if they can process batched audio and
    # backprop gradients.
    audio.requires_grad = True
    assert audio.requires_grad
    # Test batch processing
    fbank = Wav2LogFilterBank(
        sampling_rate=recording.sampling_rate,
    )
    feats = fbank(audio)
    assert feats.shape == (4, 1604, 80)
    # Test backprop
    feats.sum().backward()
    assert audio.grad is not None


def test_kaldi_mfcc_layer(recording):
    # Prepare a batch of recordings
    audio = torch.from_numpy(recording.load_audio())
    audio = torch.cat([audio] * 4, dim=0)
    assert audio.shape == (4, recording.num_samples)
    # We'll test the Kaldi feature extraction layers
    # by checking if they can process batched audio and
    # backprop gradients.
    audio.requires_grad = True
    assert audio.requires_grad
    # Test batch processing
    mfcc = Wav2MFCC(
        sampling_rate=recording.sampling_rate,
    )
    feats = mfcc(audio)
    assert feats.shape == (4, 1604, 13)
    # Test backprop
    feats.sum().backward()
    assert audio.grad is not None


def test_kaldi_spec_layer(recording):
    # Prepare a batch of recordings
    audio = torch.from_numpy(recording.load_audio())
    audio = torch.cat([audio] * 4, dim=0)
    assert audio.shape == (4, recording.num_samples)
    # We'll test the Kaldi feature extraction layers
    # by checking if they can process batched audio and
    # backprop gradients.
    audio.requires_grad = True
    assert audio.requires_grad
    # Test batch processing
    spec = Wav2Spec(
        sampling_rate=recording.sampling_rate,
    )
    feats = spec(audio)
    assert feats.shape == (4, 1604, 257)
    # Test backprop
    feats.sum().backward()
    assert audio.grad is not None


def test_kaldi_fbank_extractor(recording):
    fbank = Fbank()
    feats = fbank.extract(recording.load_audio(), recording.sampling_rate)
    assert feats.shape == (1604, 80)


def test_kaldi_fbank_extractor_vs_torchaudio(deterministic_rng, recording):
    audio = recording.load_audio()
    fbank = Fbank()
    fbank_ta = TorchaudioFbank()
    feats = fbank.extract(audio, recording.sampling_rate)
    feats_ta = fbank_ta.extract(audio, recording.sampling_rate)
    torch.testing.assert_allclose(feats, feats_ta)


def test_kaldi_mfcc_extractor(recording):
    mfcc = Mfcc()
    feats = mfcc.extract(recording.load_audio(), recording.sampling_rate)
    assert feats.shape == (1604, 13)


@pytest.mark.seed(1337)
def test_kaldi_mfcc_extractor_vs_torchaudio(deterministic_rng, recording):
    audio = recording.load_audio()
    mfcc = Mfcc()
    mfcc_ta = TorchaudioMfcc()
    feats = mfcc.extract(audio, recording.sampling_rate)
    feats_ta = mfcc_ta.extract(audio, recording.sampling_rate)
    torch.testing.assert_allclose(feats, feats_ta, rtol=1e-3, atol=1e-4)


def test_kaldi_spectrogram_extractor(recording):
    spec = Spectrogram()
    feats = spec.extract(recording.load_audio(), recording.sampling_rate)
    assert feats.shape == (1604, 257)


def test_kaldi_spectrogram_extractor_vs_torchaudio(deterministic_rng, recording):
    audio = recording.load_audio()
    spec = Spectrogram(SpectrogramConfig(use_energy=True))
    spec_ta = TorchaudioSpectrogram()
    feats = spec.extract(audio, recording.sampling_rate)
    feats_ta = spec_ta.extract(audio, recording.sampling_rate)
    # Torchaudio returns log power spectrum, while kaldi-spectrogram returns power
    # spectrum, so we need to exponentiate the torchaudio features. Also, the first
    # coefficient is the log energy, so we need to compare it separately.
    torch.testing.assert_allclose(feats[:, 0], feats_ta[:, 0])
    torch.testing.assert_allclose(feats[:, 1:], np.exp(feats_ta[:, 1:]))


@pytest.mark.parametrize(
    "extractor_type",
    [
        lambda: Fbank(FbankConfig(snip_edges=True)),
        lambda: Mfcc(MfccConfig(snip_edges=True)),
        lambda: Spectrogram(SpectrogramConfig(snip_edges=True)),
        lambda: LogSpectrogram(LogSpectrogramConfig(snip_edges=True)),
    ],
)
def test_kaldi_extractors_snip_edges_warning(extractor_type):
    with pytest.warns(UserWarning):
        extractor = extractor_type()


@pytest.mark.parametrize(
    "feature_type",
    ["kaldi-fbank", "kaldi-mfcc", "kaldi-spectrogram", "kaldi-log-spectrogram"],
)
def test_feature_extractor_serialization(feature_type):
    fe = create_default_feature_extractor(feature_type)
    with NamedTemporaryFile() as f:
        fe.to_yaml(f.name)
        fe_deserialized = type(fe).from_yaml(f.name)
    assert fe_deserialized.config == fe.config
