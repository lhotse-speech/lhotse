import multiprocessing
import sys
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from lhotse.utils import nullcontext as no_executor
from functools import partial
from tempfile import TemporaryDirectory
from unittest.mock import Mock

import pytest

from lhotse import Cut, CutSet, Fbank, LilcomHdf5Writer, Recording
from lhotse.audio import AudioSource
from lhotse.cut import MixedCut
from lhotse.features.io import LilcomFilesWriter


@pytest.fixture
def recording():
    return Recording(
        id='rec',
        sources=[AudioSource(type='file', channels=[0, 1], source='test/fixtures/stereo.wav')],
        sampling_rate=8000,
        num_samples=8000,
        duration=1.0
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
    with TemporaryDirectory() as tmpdir, LilcomFilesWriter(tmpdir) as storage:
        cut_with_feats = cut.compute_and_store_features(extractor=extractor, storage=storage)
        arr = cut_with_feats.load_features()
    assert arr.shape[0] == 100
    assert arr.shape[1] == extractor.feature_dim(cut.sampling_rate)


@pytest.mark.parametrize('mix_eagerly', [False, True])
def test_extract_and_store_features_from_mixed_cut(cut, mix_eagerly):
    mixed_cut = cut.append(cut)
    extractor = Fbank()
    with TemporaryDirectory() as tmpdir, LilcomFilesWriter(tmpdir) as storage:
        cut_with_feats = mixed_cut.compute_and_store_features(
            extractor=extractor,
            storage=storage,
            mix_eagerly=mix_eagerly
        )
        arr = cut_with_feats.load_features()
    assert arr.shape[0] == 200
    assert arr.shape[1] == extractor.feature_dim(mixed_cut.sampling_rate)


@pytest.fixture
def cut_set(cut):
    # The padding tests if feature extraction works correctly with a PaddingCut
    return CutSet.from_cuts([cut, cut.append(cut).pad(3.0)]).sort_by_duration()


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


@pytest.mark.parametrize('mix_eagerly', [False, True])
@pytest.mark.parametrize('storage_type', [LilcomFilesWriter, LilcomHdf5Writer])
@pytest.mark.parametrize(
    ['executor', 'num_jobs'],
    [
        (None, 1),
        # For some reason in tests, we need to use the "spawn" context otherwise it hangs
        pytest.param(
            partial(ProcessPoolExecutor, mp_context=multiprocessing.get_context("spawn")), 2,
            marks=pytest.mark.skipif(
                sys.version_info[0] == 3 and sys.version_info[1] < 7,
                reason="The mp_context argument is introduced in Python 3.7"
            )
        ),
        pytest.param(
            distributed.Client, 2,
            marks=pytest.mark.skipif(not is_dask_availabe(), reason='Requires Dask')
        )
    ]
)
def test_extract_and_store_features_from_cut_set(cut_set, executor, num_jobs, storage_type, mix_eagerly):
    extractor = Fbank()
    with TemporaryDirectory() as tmpdir:
        cut_set_with_feats = cut_set.compute_and_store_features(
            extractor=extractor,
            storage_path=tmpdir,
            num_jobs=num_jobs,
            mix_eagerly=mix_eagerly,
            executor=executor() if executor else None,
            storage_type=storage_type
        ).sort_by_duration()  # sort by duration to ensure the same order of cuts

        # The same number of cuts
        assert len(cut_set_with_feats) == 2

        for orig_cut, feat_cut in zip(cut_set, cut_set_with_feats):
            # The ID is retained
            assert orig_cut.id == feat_cut.id
            # Features were attached
            assert feat_cut.has_features
            # Recording is retained unless mixing a MixedCut eagerly
            should_have_recording = not (mix_eagerly and isinstance(orig_cut, MixedCut))
            assert feat_cut.has_recording == should_have_recording

        cuts = list(cut_set_with_feats)

        arr = cuts[0].load_features()
        assert arr.shape[0] == 300
        assert arr.shape[1] == extractor.feature_dim(cuts[0].sampling_rate)

        arr = cuts[1].load_features()
        assert arr.shape[0] == 100
        assert arr.shape[1] == extractor.feature_dim(cuts[0].sampling_rate)
