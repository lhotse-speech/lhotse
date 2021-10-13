import pytest
import torch

from lhotse.audio import Recording
from lhotse.features.kaldi.extractors import KaldiFbank, KaldiMfcc
from lhotse.features.kaldi.layers import Wav2LogFilterBank, Wav2MFCC


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
    fbank = Wav2MFCC(
        sampling_rate=recording.sampling_rate,
    )
    feats = fbank(audio)
    assert feats.shape == (4, 1604, 13)
    # Test backprop
    feats.sum().backward()
    assert audio.grad is not None


def test_kaldi_fbank_extractor(recording):
    fbank = KaldiFbank()
    feats = fbank.extract(recording.load_audio(), recording.sampling_rate)
    assert feats.shape == (1604, 80)


def test_kaldi_mfcc_extractor(recording):
    mfcc = KaldiMfcc()
    feats = mfcc.extract(recording.load_audio(), recording.sampling_rate)
    assert feats.shape == (1604, 13)
