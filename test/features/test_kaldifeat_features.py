from tempfile import NamedTemporaryFile

import numpy as np
import pytest
import torchaudio

from lhotse import Fbank, KaldifeatFbank
from lhotse.features import (
    create_default_feature_extractor,
)
from lhotse.utils import nullcontext as does_not_raise

# TODO: uncomment before merging
# kaldifeat = pytest.importorskip(
#     "kaldifeat", reason="Kaldifeat tests require kaldifeat to be installed."
# )


@pytest.mark.parametrize(
    ["feature_type", "exception_expectation"],
    [
        ("kaldifeat-fbank", does_not_raise()),
    ],
)
def test_feature_extractor(feature_type, exception_expectation):
    # For now, just test that it runs
    with exception_expectation:
        fe = create_default_feature_extractor(feature_type)
        samples, sr = torchaudio.load("test/fixtures/libri/libri-1088-134315-0000.wav")
        fe.extract(samples=samples, sampling_rate=sr)


@pytest.mark.parametrize(
    ["extractor1", "extractor2"],
    [
        (KaldifeatFbank(), Fbank()),
    ],
)
def test_kaldifeat_torchaudio_equivalence(extractor1, extractor2):
    sampling_rate = 16000
    np.random.seed(99)  # ensure reproducibility
    audio = np.random.rand(1, 32000).astype(np.float32)
    feat1 = extractor1.extract(audio, sampling_rate)
    feat2 = extractor2.extract(audio, sampling_rate)
    np.testing.assert_almost_equal(feat1, feat2, decimal=4)


@pytest.mark.parametrize("feature_type", ["kaldifeat-fbank"])
def test_feature_extractor_serialization(feature_type):
    fe = create_default_feature_extractor(feature_type)
    with NamedTemporaryFile() as f:
        fe.to_yaml(f.name)
        fe_deserialized = type(fe).from_yaml(f.name)
    assert fe_deserialized.config == fe.config
