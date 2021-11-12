import torch

from lhotse.features.kaldi import (
    Wav2FFT,
    Wav2LogFilterBank,
    Wav2LogSpec,
    Wav2MFCC,
    Wav2Spec,
    Wav2Win,
)


def test_wav2win():
    x = torch.randn(1, 16000, dtype=torch.float32)
    t = Wav2Win()
    y = t(x)
    assert y.shape == torch.Size([1, 100, 400])
    assert y.dtype == torch.float32


def test_wav2fft():
    x = torch.randn(1, 16000, dtype=torch.float32)
    t = Wav2FFT()
    y = t(x)
    assert y.shape == torch.Size([1, 100, 257])
    assert y.dtype == torch.complex64


def test_wav2spec():
    x = torch.randn(1, 16000, dtype=torch.float32)
    t = Wav2Spec()
    y = t(x)
    assert y.shape == torch.Size([1, 100, 257])
    assert y.dtype == torch.float32


def test_wav2logspec():
    x = torch.randn(1, 16000, dtype=torch.float32)
    t = Wav2LogSpec()
    y = t(x)
    assert y.shape == torch.Size([1, 100, 257])
    assert y.dtype == torch.float32


def test_wav2logfilterbank():
    x = torch.randn(1, 16000, dtype=torch.float32)
    t = Wav2LogFilterBank()
    y = t(x)
    assert y.shape == torch.Size([1, 100, 80])
    assert y.dtype == torch.float32


def test_wav2mfcc():
    x = torch.randn(1, 16000, dtype=torch.float32)
    t = Wav2MFCC()
    y = t(x)
    assert y.shape == torch.Size([1, 100, 13])
    assert y.dtype == torch.float32
