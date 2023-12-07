import multiprocessing
import sys
from concurrent.futures.process import ProcessPoolExecutor
from functools import partial
from tempfile import NamedTemporaryFile, TemporaryDirectory
from unittest.mock import Mock

import pytest
import torch

from lhotse import (
    S3PRLSSL,
    AudioSource,
    CutSet,
    Fbank,
    FbankConfig,
    KaldifeatFbank,
    KaldifeatMfcc,
    LibrosaFbank,
    LibrosaFbankConfig,
    LilcomChunkyWriter,
    LogSpectrogram,
    Mfcc,
    MonoCut,
    Recording,
    Spectrogram,
    SupervisionSegment,
    TorchaudioFbank,
    TorchaudioMfcc,
    load_manifest,
    validate,
)
from lhotse.cut import MixedCut
from lhotse.features.io import LilcomFilesWriter
from lhotse.serialization import InvalidPathExtension
from lhotse.utils import is_module_available
from lhotse.utils import nullcontext as does_not_raise


@pytest.fixture
def recording():
    return Recording(
        id="rec",
        sources=[
            AudioSource(type="file", channels=[0, 1], source="test/fixtures/stereo.wav")
        ],
        sampling_rate=8000,
        num_samples=8000,
        duration=1.0,
    )


@pytest.fixture
def cut(recording):
    return MonoCut(
        id="cut",
        start=0,
        duration=1.0,
        channel=0,
        recording=recording,
        supervisions=[
            SupervisionSegment(
                id="sup", recording_id=recording.id, start=0, duration=0.5
            )
        ],
    )


def test_extract_features(cut):
    extractor = Fbank(FbankConfig(sampling_rate=8000))
    arr = cut.compute_features(extractor=extractor)
    assert arr.shape[0] == 100
    assert arr.shape[1] == extractor.feature_dim(cut.sampling_rate)


def test_extract_and_store_features(cut):
    extractor = Fbank(FbankConfig(sampling_rate=8000))
    with TemporaryDirectory() as tmpdir, LilcomFilesWriter(tmpdir) as storage:
        cut_with_feats = cut.compute_and_store_features(
            extractor=extractor, storage=storage
        )
        arr = cut_with_feats.load_features()
    assert arr.shape[0] == 100
    assert arr.shape[1] == extractor.feature_dim(cut.sampling_rate)


@pytest.mark.parametrize("mix_eagerly", [False, True])
def test_extract_and_store_features_from_mixed_cut(cut, mix_eagerly):
    mixed_cut = cut.append(cut)
    extractor = Fbank(FbankConfig(sampling_rate=8000))
    with TemporaryDirectory() as tmpdir, LilcomFilesWriter(tmpdir) as storage:
        cut_with_feats = mixed_cut.compute_and_store_features(
            extractor=extractor, storage=storage, mix_eagerly=mix_eagerly
        )
        validate(cut_with_feats)
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


@pytest.mark.parametrize("mix_eagerly", [False, True])
@pytest.mark.parametrize("storage_type", [LilcomFilesWriter, LilcomChunkyWriter])
@pytest.mark.parametrize(
    ["executor", "num_jobs"],
    [
        (None, 1),
        # For some reason in tests, we need to use the "spawn" context otherwise it hangs
        pytest.param(
            partial(
                ProcessPoolExecutor, mp_context=multiprocessing.get_context("spawn")
            ),
            2,
            marks=pytest.mark.skipif(
                sys.version_info[0] == 3 and sys.version_info[1] < 7,
                reason="The mp_context argument is introduced in Python 3.7",
            ),
        ),
        pytest.param(
            distributed.Client,
            2,
            marks=pytest.mark.skipif(not is_dask_availabe(), reason="Requires Dask"),
        ),
    ],
)
def test_extract_and_store_features_from_cut_set(
    cut_set, executor, num_jobs, storage_type, mix_eagerly
):
    extractor = Fbank(FbankConfig(sampling_rate=8000))
    with TemporaryDirectory() as tmpdir:
        cut_set_with_feats = cut_set.compute_and_store_features(
            extractor=extractor,
            storage_path=tmpdir,
            num_jobs=num_jobs,
            mix_eagerly=mix_eagerly,
            executor=executor() if executor else None,
            storage_type=storage_type,
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


def is_python_311_or_higher() -> bool:
    import sys

    return sys.version_info[:2] > (3, 10)


@pytest.mark.parametrize(
    "extractor_type",
    [
        Fbank,
        Mfcc,
        Spectrogram,
        LogSpectrogram,
        TorchaudioFbank,
        TorchaudioMfcc,
        pytest.param(
            KaldifeatFbank,
            marks=pytest.mark.skipif(
                not is_module_available("kaldifeat"),
                reason="Requires kaldifeat to run.",
            ),
        ),
        pytest.param(
            KaldifeatMfcc,
            marks=pytest.mark.skipif(
                not is_module_available("kaldifeat"),
                reason="Requires kaldifeat to run.",
            ),
        ),
        pytest.param(
            lambda: LibrosaFbank(LibrosaFbankConfig(sampling_rate=16000)),
            marks=[
                pytest.mark.skipif(
                    not is_module_available("librosa"),
                    reason="Requires librosa to run.",
                ),
            ],
        ),
        pytest.param(
            S3PRLSSL,
            marks=[
                pytest.mark.skipif(
                    not is_module_available("s3prl") or is_python_311_or_higher(),
                    reason="Requires s3prl to run.",
                ),
            ],
        ),
    ],
)
def test_cut_set_batch_feature_extraction(cut_set, extractor_type):
    extractor = extractor_type()
    cut_set = cut_set.resample(16000)
    with NamedTemporaryFile() as tmpf:
        cut_set_with_feats = cut_set.compute_and_store_features_batch(
            extractor=extractor,
            storage_path=tmpf.name,
            num_workers=0,
        )
        validate(cut_set_with_feats, read_data=True)


@pytest.mark.parametrize(
    "extractor_type",
    [
        Fbank,
        TorchaudioFbank,
        pytest.param(
            KaldifeatFbank,
            marks=pytest.mark.skipif(
                not is_module_available("kaldifeat"),
                reason="Requires kaldifeat to run.",
            ),
        ),
        pytest.param(
            S3PRLSSL,
            marks=[
                pytest.mark.skipif(
                    not is_module_available("s3prl") or is_python_311_or_higher(),
                    reason="Requires s3prl to run.",
                ),
            ],
        ),
    ],
)
def test_cut_set_batch_feature_extraction_with_collation(cut_set, extractor_type):
    extractor = extractor_type()
    cut_set = cut_set.resample(16000)
    with NamedTemporaryFile() as tmpf:
        cut_set_with_feats = cut_set.compute_and_store_features_batch(
            extractor=extractor,
            storage_path=tmpf.name,
            num_workers=0,
            collate=True,
        )
        validate(cut_set_with_feats, read_data=True)


@pytest.mark.parametrize(
    ["suffix", "exception_expectation"],
    [
        (".jsonl", does_not_raise()),
        (".json", pytest.raises(InvalidPathExtension)),
    ],
)
def test_cut_set_batch_feature_extraction_manifest_path(
    cut_set, suffix, exception_expectation
):
    extractor = Fbank()
    cut_set = cut_set.resample(16000)
    with NamedTemporaryFile() as feat_f, NamedTemporaryFile(
        suffix=suffix
    ) as manifest_f:
        with exception_expectation:
            cut_set_with_feats = cut_set.compute_and_store_features_batch(
                extractor=extractor,
                storage_path=feat_f.name,
                manifest_path=manifest_f.name,
                num_workers=0,
            )
            validate(cut_set_with_feats, read_data=True)


@pytest.mark.parametrize("overwrite", [False, True])
def test_cut_set_batch_feature_extraction_resume(cut_set, overwrite):
    # This test checks that we can keep writing to the same file
    # and the previously written results are not lost.
    # Since we don't have an easy way to interrupt the execution in a test,
    # we just write another CutSet to the same file.
    # The effect is the same.
    extractor = Fbank()
    cut_set = cut_set.resample(16000)
    subsets = cut_set.split(num_splits=2)
    processed = []
    with NamedTemporaryFile() as feat_f, NamedTemporaryFile(
        suffix=".jsonl.gz"
    ) as manifest_f:
        for cuts in subsets:
            processed.append(
                cuts.compute_and_store_features_batch(
                    extractor=extractor,
                    storage_path=feat_f.name,
                    manifest_path=manifest_f.name,
                    num_workers=0,
                    overwrite=overwrite,
                )
            )
        feat_f.flush()
        manifest_f.flush()
        merged = load_manifest(manifest_f.name)
        if overwrite:
            assert list(merged.ids) == list(subsets[-1].ids)
        else:
            assert list(merged.ids) == list(cut_set.ids)
        validate(merged, read_data=True)


@pytest.mark.parametrize(
    "extractor_type",
    [
        Fbank,
        Mfcc,
        TorchaudioFbank,
        TorchaudioMfcc,
        pytest.param(
            KaldifeatFbank,
            marks=pytest.mark.skipif(
                not is_module_available("kaldifeat"),
                reason="Requires kaldifeat to run.",
            ),
        ),
        pytest.param(
            KaldifeatMfcc,
            marks=pytest.mark.skipif(
                not is_module_available("kaldifeat"),
                reason="Requires kaldifeat to run.",
            ),
        ),
        pytest.param(
            lambda: LibrosaFbank(LibrosaFbankConfig(sampling_rate=16000)),
            marks=[
                pytest.mark.skipif(
                    not is_module_available("librosa"),
                    reason="Requires librosa to run.",
                ),
            ],
        ),
    ],
)
def test_on_the_fly_batch_feature_extraction(cut_set, extractor_type):
    from lhotse.dataset import OnTheFlyFeatures

    extractor = OnTheFlyFeatures(extractor=extractor_type())
    cut_set = cut_set.resample(16000)
    feats, feat_lens = extractor(cut_set)  # does not crash
    assert isinstance(feats, torch.Tensor)
    assert isinstance(feat_lens, torch.Tensor)


def test_on_the_fly_feats_return_audio(cut_set):
    from lhotse.dataset import OnTheFlyFeatures

    extractor = OnTheFlyFeatures(extractor=Fbank(), return_audio=True)
    cut_set = cut_set.resample(16000)
    feats, feat_lens, audios, audio_lens = extractor(cut_set)
    assert isinstance(feats, torch.Tensor)
    assert isinstance(feat_lens, torch.Tensor)
    assert isinstance(audios, torch.Tensor)
    assert isinstance(audio_lens, torch.Tensor)

    assert feats.shape == (2, 300, 80)
    assert feat_lens.shape == (2,)
    assert audios.shape == (2, 48000)
    assert audio_lens.shape == (2,)
