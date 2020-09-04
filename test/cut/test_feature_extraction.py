from tempfile import TemporaryDirectory
from unittest.mock import Mock

import pytest
import numpy as np

from lhotse import Recording, Cut, Fbank


@pytest.fixture
def audio_source():
    # Return a mocked "AudioSource" object that loads a 1s long 1000Hz sine wave
    source = Mock()
    source.load_audio = Mock(return_value=np.sin(2 * np.pi * 1000 * np.arange(0, 8000, dtype=np.float32)))
    source.channels = [0]
    return source


@pytest.fixture
def recording(audio_source):
    return Recording(
        id='rec', sources=[audio_source], sampling_rate=8000, num_samples=8000, duration=1.0
    )


@pytest.fixture
def cut(recording):
    return Cut(id='cut', start=0, duration=1.0, channel=0, recording=recording)


def test_extract_features(cut):
    extractor = Fbank()
    arr = cut.compute_features(extractor=extractor)
    assert arr.shape[0] == 100
    assert arr.shape[1] == extractor.feature_dim(cut.sampling_rate)


def test_extract_and_store_features(cut):
    extractor = Fbank()
    with TemporaryDirectory() as tmpdir:
        cut.compute_and_store_features(extractor=extractor, output_dir=tmpdir)
        arr = cut.load_features()
    assert arr.shape[0] == 100
    assert arr.shape[1] == extractor.feature_dim(cut.sampling_rate)
