from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from contextlib import nullcontext
from tempfile import TemporaryDirectory
from unittest.mock import Mock

import pytest
import numpy as np

from lhotse import Recording, Cut, Fbank, CutSet


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
        cut_with_feats = cut.compute_and_store_features(extractor=extractor, output_dir=tmpdir)
        arr = cut_with_feats.load_features()
    assert arr.shape[0] == 100
    assert arr.shape[1] == extractor.feature_dim(cut.sampling_rate)


@pytest.mark.parametrize('mix_eagerly', [False, True])
def test_extract_and_store_features_from_mixed_cut(cut, mix_eagerly):
    mixed_cut = cut.append(cut)
    extractor = Fbank()
    with TemporaryDirectory() as tmpdir:
        cut_with_feats = mixed_cut.compute_and_store_features(
            extractor=extractor,
            output_dir=tmpdir,
            mix_eagerly=mix_eagerly
        )
        arr = cut_with_feats.load_features()
    assert arr.shape[0] == 200
    assert arr.shape[1] == extractor.feature_dim(mixed_cut.sampling_rate)


@pytest.fixture
def cut_set(cut):
    return CutSet.from_cuts([cut, cut.append(cut)])


# The lines below try to import Dask (a distributed computing library for Python)
# so that we can test that parallel feature extraction through the "executor"
# interface works correctly in this case. Dask in not a requirement or a dependency
# of Lhotse, so we make the tests with it optional as well.

try:
    import distributed
except:
    distributed = Mock()


def is_dask_availabe():
    try:
        import dask
        import distributed
        return True
    except:
        return False


@pytest.mark.parametrize(
    'mix_eagerly', [False, True]
)
@pytest.mark.parametrize(
    'executor', [
        nullcontext(),
        ThreadPoolExecutor(2),
        ProcessPoolExecutor(2),
        pytest.param(distributed.Client(), marks=pytest.mark.skipif(not is_dask_availabe(), reason='Requires Dask'))
    ]
)
def test_extract_and_store_features_from_cut_set(cut_set, executor, mix_eagerly):
    extractor = Fbank()
    with executor, TemporaryDirectory() as tmpdir:
        cut_set_with_feats = cut_set.compute_and_store_features(
            extractor=extractor,
            output_dir=tmpdir,
            mix_eagerly=mix_eagerly
        )
        assert len(cut_set_with_feats) == 2
        cuts = list(cut_set_with_feats)
        arr = cuts[0].load_features()
        assert arr.shape[0] == 100
        assert arr.shape[1] == extractor.feature_dim(cuts[0].sampling_rate)
        arr = cuts[1].load_features()
        assert arr.shape[0] == 200
        assert arr.shape[1] == extractor.feature_dim(cuts[0].sampling_rate)
