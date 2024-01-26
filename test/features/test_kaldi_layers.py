import math

import pytest
import torch

from lhotse.features.kaldi import (
    Wav2FFT,
    Wav2LogFilterBank,
    Wav2LogSpec,
    Wav2MFCC,
    Wav2Spec,
    Wav2Win,
)
from lhotse.features.kaldi.layers import (
    _get_strided_batch,
    _get_strided_batch_streaming,
)
from lhotse.testing.random import deterministic_rng


def test_wav2win(deterministic_rng):
    x = torch.randn(1, 16000, dtype=torch.float32)
    t = Wav2Win()
    y, _ = t(x)
    assert y.shape == torch.Size([1, 100, 400])
    assert y.dtype == torch.float32


def test_wav2fft(deterministic_rng):
    x = torch.randn(1, 16000, dtype=torch.float32)
    t = Wav2FFT()
    y = t(x)
    assert y.shape == torch.Size([1, 100, 257])
    assert y.dtype == torch.complex64


def test_wav2spec(deterministic_rng):
    x = torch.randn(1, 16000, dtype=torch.float32)
    t = Wav2Spec()
    y = t(x)
    assert y.shape == torch.Size([1, 100, 257])
    assert y.dtype == torch.float32


def test_wav2logspec(deterministic_rng):
    x = torch.randn(1, 16000, dtype=torch.float32)
    t = Wav2LogSpec()
    y = t(x)
    assert y.shape == torch.Size([1, 100, 257])
    assert y.dtype == torch.float32


def test_wav2logfilterbank(deterministic_rng):
    x = torch.randn(1, 16000, dtype=torch.float32)
    t = Wav2LogFilterBank()
    y = t(x)
    assert y.shape == torch.Size([1, 100, 80])
    assert y.dtype == torch.float32


def test_wav2mfcc(deterministic_rng):
    x = torch.randn(1, 16000, dtype=torch.float32)
    t = Wav2MFCC()
    y = t(x)
    assert y.shape == torch.Size([1, 100, 13])
    assert y.dtype == torch.float32


def test_wav2win_is_torchscriptable(deterministic_rng):
    x = torch.randn(1, 16000, dtype=torch.float32)
    t = torch.jit.script(Wav2Win())
    y, _ = t(x)
    assert y.shape == torch.Size([1, 100, 400])
    assert y.dtype == torch.float32


def test_wav2fft_is_torchscriptable(deterministic_rng):
    x = torch.randn(1, 16000, dtype=torch.float32)
    t = torch.jit.script(Wav2FFT())
    y = t(x)
    assert y.shape == torch.Size([1, 100, 257])
    assert y.dtype == torch.complex64


def test_wav2spec_is_torchscriptable(deterministic_rng):
    x = torch.randn(1, 16000, dtype=torch.float32)
    t = torch.jit.script(Wav2Spec())
    y = t(x)
    assert y.shape == torch.Size([1, 100, 257])
    assert y.dtype == torch.float32


def test_wav2logspec_is_torchscriptable(deterministic_rng):
    x = torch.randn(1, 16000, dtype=torch.float32)
    t = torch.jit.script(Wav2LogSpec())
    y = t(x)
    assert y.shape == torch.Size([1, 100, 257])
    assert y.dtype == torch.float32


def test_wav2logfilterbank_is_torchscriptable(deterministic_rng):
    x = torch.randn(1, 16000, dtype=torch.float32)
    t = torch.jit.script(Wav2LogFilterBank())
    y = t(x)
    assert y.shape == torch.Size([1, 100, 80])
    assert y.dtype == torch.float32


def test_wav2mfcc_is_torchscriptable(deterministic_rng):
    x = torch.randn(1, 16000, dtype=torch.float32)
    t = torch.jit.script(Wav2MFCC())
    y = t(x)
    assert y.shape == torch.Size([1, 100, 13])
    assert y.dtype == torch.float32


def test_strided_waveform_batch_streaming_snip_edges_false(deterministic_rng):
    x = torch.arange(16000).unsqueeze(0)
    window_length = 400
    window_shift = 160

    # reference offline forward pass
    y = _get_strided_batch(
        waveform=x,
        window_length=window_length,
        window_shift=window_shift,
        snip_edges=False,
    )
    assert y.shape == torch.Size([1, 100, window_length])

    # online pass under test
    frames = []
    chunk_size = 1200
    num_chunks = math.ceil(x.size(1) / chunk_size)
    remainder = None
    for i in range(num_chunks):
        x_chunk = x[:, i * chunk_size : (i + 1) * chunk_size]
        y_chunk, remainder = _get_strided_batch_streaming(
            x_chunk, window_length=400, window_shift=160, prev_remainder=remainder
        )
        frames.append(y_chunk)

    # one last extra iteration to pump out the frames that the offline version gets due to padding
    # note that our implementation will flip the last N samples so we do the same for identical results
    x_chunk = torch.flip(x[:, -window_shift:], dims=(1,))
    y_chunk, remainder = _get_strided_batch_streaming(
        x_chunk,
        window_length=window_length,
        window_shift=window_shift,
        prev_remainder=remainder,
        snip_edges=False,
    )
    frames.append(y_chunk)

    y_online = torch.cat(frames, dim=1)

    assert y.shape == y_online.shape

    torch.testing.assert_allclose(y_online, y)


def test_strided_waveform_batch_streaming_snip_edges_true(deterministic_rng):
    x = torch.arange(16000).unsqueeze(0)
    window_length = 400
    window_shift = 160

    # reference offline forward pass
    y = _get_strided_batch(
        waveform=x,
        window_length=window_length,
        window_shift=window_shift,
        snip_edges=True,
    )
    assert y.shape == torch.Size([1, 98, window_length])

    # online pass under test
    frames = []
    chunk_size = 1200
    num_chunks = math.ceil(x.size(1) / chunk_size)
    remainder = None
    for i in range(num_chunks):
        x_chunk = x[:, i * chunk_size : (i + 1) * chunk_size]
        y_chunk, remainder = _get_strided_batch_streaming(
            x_chunk,
            window_length=400,
            window_shift=160,
            prev_remainder=remainder,
            snip_edges=True,
        )
        frames.append(y_chunk)

    # Note: unlike the case of snip_edges=False,
    # we don't run one last step with reflected samples here
    # to match the behavior of offline snip_edges=True.

    y_online = torch.cat(frames, dim=1)

    assert y.shape == y_online.shape

    torch.testing.assert_allclose(y_online, y)


def test_wav2win_streaming(deterministic_rng):
    x = torch.randn(1, 16000, dtype=torch.float32)
    t = Wav2Win()
    window_length = 400
    window_shift = 160
    assert t._length == window_length
    assert t._shift == window_shift

    # reference offline forward pass
    y, _ = t(x)
    assert y.shape == torch.Size([1, 100, window_length])

    # online pass under test
    frames = []
    chunk_size = 1200
    num_chunks = math.ceil(x.size(1) / chunk_size)
    remainder = None
    for i in range(num_chunks):
        x_chunk = x[:, i * chunk_size : (i + 1) * chunk_size]
        (y_chunk, _), remainder = t.online_inference(x_chunk, context=remainder)
        frames.append(y_chunk)

    # one last extra iteration to pump out the frames that the offline version gets due to padding
    x_chunk = torch.flip(x[:, -window_shift:], dims=(1,))
    (y_chunk, _), remainder = t.online_inference(x_chunk, context=remainder)
    frames.append(y_chunk)

    y_online = torch.cat(frames, dim=1)

    assert y.shape == y_online.shape

    torch.testing.assert_allclose(y_online, y)


@pytest.mark.parametrize(
    ["layer_type", "feat_dim"],
    [
        (Wav2FFT, 257),
        (Wav2Spec, 257),
        (Wav2LogSpec, 257),
        (Wav2LogFilterBank, 80),
        (Wav2MFCC, 13),
    ],
)
def test_wav2logfilterbank_streaming(deterministic_rng, layer_type, feat_dim):
    x = torch.randn(1, 16000, dtype=torch.float32)
    t = layer_type()
    window_length = 400
    window_shift = 160
    assert t.wav2win._length == window_length
    assert t.wav2win._shift == window_shift

    # reference offline forward pass
    y = t(x)
    assert y.shape == torch.Size([1, 100, feat_dim])

    # online pass under test
    frames = []
    chunk_size = 1200
    num_chunks = math.ceil(x.size(1) / chunk_size)
    remainder = None
    for i in range(num_chunks):
        x_chunk = x[:, i * chunk_size : (i + 1) * chunk_size]
        y_chunk, remainder = t.online_inference(x_chunk, context=remainder)
        frames.append(y_chunk)

    # one last extra iteration to pump out the frames that the offline version gets due to padding
    x_chunk = torch.flip(x[:, -window_shift:], dims=(1,))
    y_chunk, remainder = t.online_inference(x_chunk, context=remainder)
    frames.append(y_chunk)

    y_online = torch.cat(frames, dim=1)

    assert y.shape == y_online.shape

    torch.testing.assert_allclose(y_online, y)
