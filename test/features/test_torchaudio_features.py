from tempfile import NamedTemporaryFile

import pytest
import torchaudio

from lhotse import Fbank, FeatureExtractor, create_default_feature_extractor
from lhotse.utils import nullcontext as does_not_raise


@pytest.mark.parametrize(
    ["feature_type", "exception_expectation"],
    [
        ("mfcc", does_not_raise()),
        ("fbank", does_not_raise()),
        ("spectrogram", does_not_raise()),
        ("pitch", pytest.raises(Exception)),
    ],
)
def test_feature_extractor(feature_type, exception_expectation):
    # For now, just test that it runs
    with exception_expectation:
        fe = create_default_feature_extractor(feature_type)
        samples, sr = torchaudio.load("test/fixtures/libri/libri-1088-134315-0000.wav")
        fe.extract(samples=samples, sampling_rate=sr)


@pytest.mark.parametrize(
    ["feature_type", "exception_expectation"],
    [
        ("mfcc", does_not_raise()),
        ("fbank", does_not_raise()),
        ("spectrogram", does_not_raise()),
        ("pitch", pytest.raises(Exception)),
    ],
)
def test_feature_extractor_batch_extract_uneven_sequences(feature_type, exception_expectation):
    # For now, just test that it runs
    with exception_expectation:
        fe = create_default_feature_extractor(feature_type)
        samples, sr = torchaudio.load("test/fixtures/libri/libri-1088-134315-0000.wav")
        results = fe.extract_batch(samples=[samples, samples[:, :4000]], sampling_rate=sr)
        assert isinstance(results, list)
        assert len(results) == 2


def test_feature_extractor_serialization():
    fe = Fbank()
    with NamedTemporaryFile() as f:
        fe.to_yaml(f.name)
        fe_deserialized = Fbank.from_yaml(f.name)
    assert fe_deserialized.config == fe.config


def test_feature_extractor_generic_deserialization():
    fe = Fbank()
    with NamedTemporaryFile() as f:
        fe.to_yaml(f.name)
        fe_deserialized = FeatureExtractor.from_yaml(f.name)
    assert fe_deserialized.config == fe.config
