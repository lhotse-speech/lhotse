import numpy as np
import pytest

from lhotse import S3PRLSSL, Recording, S3PRLSSLConfig
from lhotse.utils import compute_num_frames, compute_num_samples, is_module_available


@pytest.fixture()
def recording():
    return Recording.from_file("test/fixtures/libri/libri-1088-134315-0000.wav")


@pytest.mark.skipif(
    not is_module_available("s3prl.hub"), reason="The test requires s3prl to run."
)
def test_s3prl_feature_extractor_default_config(recording):
    feature_extractor = S3PRLSSL()
    y = feature_extractor.extract(recording.load_audio(), recording.sampling_rate)
    assert np.shape(y) == (802, 1024)


@pytest.mark.skipif(
    not is_module_available("s3prl.hub"), reason="The test requires s3prl to run."
)
def test_s3prl_feature_extractor_config(recording):
    config = S3PRLSSLConfig(
        ssl_model="wav2vec2",
        feature_dim=768,
    )
    feature_extractor = S3PRLSSL(config=config)
    y = feature_extractor.extract(recording.load_audio(), recording.sampling_rate)
    assert np.shape(y) == (802, 768)
