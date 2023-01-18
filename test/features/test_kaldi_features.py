import numpy as np
import pytest
import torch

from lhotse import TorchaudioFbank, TorchaudioMfcc, TorchaudioSpectrogram
from lhotse.audio import Recording
from lhotse.features.kaldi.extractors import Fbank, Mfcc, Spectrogram, SpectrogramConfig
from lhotse.features.kaldi.layers import Wav2LogFilterBank, Wav2MFCC, Wav2Spec


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


def test_kaldi_fbank_extractor_vs_torchaudio(recording):
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


def test_kaldi_mfcc_extractor_vs_torchaudio(recording):
    audio = recording.load_audio()
    mfcc = Mfcc()
    mfcc_ta = TorchaudioMfcc()
    feats = mfcc.extract(audio, recording.sampling_rate)
    feats_ta = mfcc_ta.extract(audio, recording.sampling_rate)
    torch.testing.assert_allclose(feats, feats_ta)


def test_kaldi_spectrogram_extractor(recording):
    spec = Spectrogram()
    feats = spec.extract(recording.load_audio(), recording.sampling_rate)
    assert feats.shape == (1604, 257)


def test_kaldi_spectrogram_extractor_vs_torchaudio(recording):
    audio = recording.load_audio()
    spec = Spectrogram()
    spec_ta = TorchaudioSpectrogram()
    feats = spec.extract(audio, recording.sampling_rate)
    feats_ta = spec_ta.extract(audio, recording.sampling_rate)
    # Torchaudio returns log power spectrum, while kaldi-spectrogram returns power
    # spectrum, so we need to take log
    torch.testing.assert_allclose(np.log(feats), feats_ta)
