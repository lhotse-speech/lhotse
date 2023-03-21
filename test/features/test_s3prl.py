import sys

import numpy as np
import pytest
import torch

from lhotse import S3PRLSSL, Recording, S3PRLSSLConfig

s3prl = pytest.importorskip("s3prl", reason="The test requires s3prl to run.")


if sys.version_info[:2] > (3, 10):
    pytest.skip(
        "S3PRL does not support Python 3.11 as of 21-Mar-2023.", allow_module_level=True
    )


@pytest.fixture()
def recording():
    return Recording.from_file("test/fixtures/libri/libri-1088-134315-0000.wav")


def test_s3prl_feature_extractor_default_config(recording):
    feature_extractor = S3PRLSSL()
    y = feature_extractor.extract(recording.load_audio(), recording.sampling_rate)
    assert np.shape(y) == (802, 1024)


def test_s3prl_feature_extractor_config(recording):
    config = S3PRLSSLConfig(
        ssl_model="wav2vec2",
        feature_dim=768,
    )
    feature_extractor = S3PRLSSL(config=config)
    y = feature_extractor.extract(recording.load_audio(), recording.sampling_rate)
    assert np.shape(y) == (802, 768)


@pytest.mark.parametrize(
    "input",
    [
        np.arange(8000, dtype=np.float32),
        torch.arange(8000, dtype=torch.float32),
        torch.arange(8000, dtype=torch.float32),
    ],
)
def test_s3prl_supports_single_input_waveform(input):
    config = S3PRLSSLConfig(
        ssl_model="wav2vec2",
        feature_dim=768,
    )
    fe = S3PRLSSL(config=config)
    feats = fe.extract(input, sampling_rate=16000)
    assert feats.shape == (25, 768)


@pytest.mark.parametrize(
    "input",
    [
        [np.arange(8000, dtype=np.float32)],
        [np.arange(8000, dtype=np.float32).reshape(1, -1)],
        [torch.arange(8000, dtype=torch.float32).unsqueeze(0)],
    ],
)
def test_s3prl_supports_list_with_single_input_waveform(input):
    config = S3PRLSSLConfig(
        ssl_model="wav2vec2",
        feature_dim=768,
    )
    fe = S3PRLSSL(config=config)
    feats = fe.extract(input, sampling_rate=16000)
    assert isinstance(feats, list)
    assert len(feats) == 1
    assert feats[0].shape == (25, 768)


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
def test_s3prl_supports_list_of_even_len_inputs(input):
    config = S3PRLSSLConfig(
        ssl_model="wav2vec2",
        feature_dim=768,
    )
    fe = S3PRLSSL(config=config)
    feats = fe.extract(input, sampling_rate=16000)
    assert feats.ndim == 3
    assert feats.shape == (2, 25, 768)


def test_s3prl_supports_list_of_uneven_len_inputs():
    input = [
        torch.arange(8000, dtype=torch.float32),
        torch.arange(16000, dtype=torch.float32),
    ]
    config = S3PRLSSLConfig(
        ssl_model="wav2vec2",
        feature_dim=768,
    )
    fe = S3PRLSSL(config=config)
    feats = fe.extract(input, sampling_rate=16000)
    assert len(feats) == 2
    f1, f2 = feats
    assert f1.shape == (25, 768)
    assert f2.shape == (50, 768)
