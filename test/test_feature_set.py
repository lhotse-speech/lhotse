import pickle
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Optional

import numpy as np
import pytest
from pytest import mark, raises

from lhotse import FbankConfig, MfccConfig
from lhotse.audio import RecordingSet
from lhotse.features import (
    Fbank,
    FeatureMixer,
    Features,
    FeatureSet,
    FeatureSetBuilder,
    Mfcc,
    Spectrogram,
)
from lhotse.features.io import (
    ChunkedLilcomHdf5Writer,
    KaldiWriter,
    LilcomChunkyWriter,
    LilcomFilesWriter,
    LilcomHdf5Writer,
    MemoryLilcomWriter,
    NumpyFilesWriter,
    NumpyHdf5Writer,
)
from lhotse.testing.dummies import DummyManifest
from lhotse.utils import Seconds, is_module_available
from lhotse.utils import nullcontext as does_not_raise
from lhotse.utils import time_diff_to_num_frames

other_params = {}
some_augmentation = None


@mark.parametrize(
    [
        "recording_id",
        "channel",
        "start",
        "duration",
        "exception_expectation",
        "expected_num_frames",
    ],
    [
        ("recording-1", 0, 0.0, None, does_not_raise(), 50),  # whole recording
        (
            "recording-1",
            0,
            0.0,
            0.499,
            does_not_raise(),
            50,
        ),  # practically whole recording
        ("recording-2", 0, 0.0, 0.7, does_not_raise(), 70),
        ("recording-2", 0, 0.5, 0.5, does_not_raise(), 50),
        ("recording-2", 1, 0.25, 0.65, does_not_raise(), 65),
        ("recording-nonexistent", 0, 0.0, None, raises(KeyError), None),  # no recording
        ("recording-1", 1000, 0.0, None, raises(KeyError), None),  # no channel
        (
            "recording-2",
            0,
            0.5,
            1.0,
            raises(KeyError),
            None,
        ),  # no features between [1.0, 1.5]
        ("recording-2", 0, 1.5, None, raises(KeyError), None),  # no features after 1.0
    ],
)
def test_load_features(
    recording_id: str,
    channel: int,
    start: float,
    duration: float,
    exception_expectation,
    expected_num_frames: Optional[float],
):
    # just test that it loads
    feature_set = FeatureSet.from_json(
        "test/fixtures/dummy_feats/feature_manifest.json"
    )
    with exception_expectation:
        features = feature_set.load(
            recording_id, channel_id=channel, start=start, duration=duration
        )
        # expect a matrix
        assert len(features.shape) == 2
        # expect time as the first dimension
        assert features.shape[0] == expected_num_frames


def test_load_features_with_default_arguments():
    feature_set = FeatureSet.from_json(
        "test/fixtures/dummy_feats/feature_manifest.json"
    )
    features = feature_set.load("recording-1")
    assert features.shape == (50, 23)


def test_compute_global_stats():
    feature_set = FeatureSet.from_json(
        "test/fixtures/dummy_feats/feature_manifest.json"
    )
    with NamedTemporaryFile() as f:
        stats = feature_set.compute_global_stats(storage_path=f.name)
        f.flush()
        read_stats = pickle.load(f)
    # Post-condition 1: feature dim is consistent
    assert stats["norm_means"].shape == (feature_set[0].num_features,)
    assert stats["norm_stds"].shape == (feature_set[0].num_features,)
    # Post-condition 2: the iterative method yields very close results to
    # the "standard" method.
    true_means = np.mean(np.concatenate([f.load() for f in feature_set]), axis=0)
    true_stds = np.std(np.concatenate([f.load() for f in feature_set]), axis=0)
    np.testing.assert_almost_equal(stats["norm_means"], true_means, decimal=5)
    np.testing.assert_almost_equal(stats["norm_stds"], true_stds, decimal=5)
    # Post-condition 3: the serialization works correctly
    assert (stats["norm_means"] == read_stats["norm_means"]).all()
    assert (stats["norm_stds"] == read_stats["norm_stds"]).all()


@pytest.mark.parametrize(
    "storage_fn",
    [
        lambda: LilcomFilesWriter(TemporaryDirectory().name),
        lambda: LilcomChunkyWriter(NamedTemporaryFile().name),
        lambda: NumpyFilesWriter(TemporaryDirectory().name),
        lambda: MemoryLilcomWriter(),
        pytest.param(
            lambda: NumpyHdf5Writer(NamedTemporaryFile().name),
            marks=pytest.mark.skipif(
                not is_module_available("h5py"),
                reason="h5py must be installed for HDF5 writing",
            ),
        ),
        pytest.param(
            lambda: LilcomHdf5Writer(NamedTemporaryFile().name),
            marks=pytest.mark.skipif(
                not is_module_available("h5py"),
                reason="h5py must be installed for HDF5 writing",
            ),
        ),
        pytest.param(
            lambda: ChunkedLilcomHdf5Writer(NamedTemporaryFile().name),
            marks=pytest.mark.skipif(
                not is_module_available("h5py"),
                reason="h5py must be installed for HDF5 writing",
            ),
        ),
        pytest.param(
            lambda: KaldiWriter(TemporaryDirectory().name),
            marks=pytest.mark.skipif(
                not is_module_available("kaldi_native_io"),
                reason="kaldi_native_io must be installed for scp+ark feature writing",
            ),
        ),
    ],
)
def test_feature_set_builder(storage_fn):
    recordings: RecordingSet = RecordingSet.from_json("test/fixtures/audio.json")
    extractor = Fbank(FbankConfig(sampling_rate=8000))
    with storage_fn() as storage:
        builder = FeatureSetBuilder(
            feature_extractor=extractor,
            storage=storage,
        )
        feature_set = builder.process_and_store_recordings(recordings=recordings)

    assert len(feature_set) == 6

    feature_infos = list(feature_set)

    # Assert the properties shared by all features
    for features in feature_infos:
        # assert that fbank is the default feature type
        assert features.type == "kaldi-fbank"
        # assert that duration is always a multiple of frame_shift
        assert features.num_frames == round(features.duration / features.frame_shift)
        # assert that num_features is preserved
        assert features.num_features == builder.feature_extractor.config.num_filters
        # assert that the storage type metadata matches
        assert features.storage_type == storage.name
        # assert that the metadata is consistent with the data shapes
        arr = features.load()
        assert arr.shape[0] == features.num_frames
        assert arr.shape[1] == features.num_features
        # assert that the stored features are the same as the "freshly extracted" features
        recording = recordings[features.recording_id]
        expected = extractor.extract(
            samples=recording.load_audio(channels=features.channels),
            sampling_rate=recording.sampling_rate,
        )
        np.testing.assert_almost_equal(arr, expected, decimal=2)

    # Assert the properties for recordings of duration 0.5 seconds
    for features in feature_infos[:2]:
        assert features.num_frames == 50
        assert features.duration == 0.5

    # Assert the properties for recordings of duration 1.0 seconds
    for features in feature_infos[2:]:
        assert features.num_frames == 100
        assert features.duration == 1.0


@mark.parametrize(
    ["time_diff", "frame_length", "frame_shift", "expected_num_frames"],
    [
        (1.0, 0.025, 0.01, 98),
        (0.5, 0.025, 0.01, 48),
        (1.0, 0.025, 0.012, 82),
    ],
)
def test_time_diff_to_num_frames(
    time_diff: Seconds,
    frame_length: Seconds,
    frame_shift: Seconds,
    expected_num_frames: int,
):
    assert (
        time_diff_to_num_frames(
            time_diff=time_diff, frame_length=frame_length, frame_shift=frame_shift
        )
        == expected_num_frames
    )


def test_add_feature_sets():
    expected = DummyManifest(FeatureSet, begin_id=0, end_id=10)
    feature_set_1 = DummyManifest(FeatureSet, begin_id=0, end_id=5)
    feature_set_2 = DummyManifest(FeatureSet, begin_id=5, end_id=10)
    combined = feature_set_1 + feature_set_2
    assert combined.to_eager() == expected


@pytest.mark.parametrize(
    ["feature_extractor", "decimal", "exception_expectation"],
    [
        (Fbank(FbankConfig(num_filters=40, sampling_rate=8000)), 0, does_not_raise()),
        (Spectrogram(), -1, does_not_raise()),
        (Mfcc(MfccConfig(sampling_rate=8000)), None, raises(ValueError)),
    ],
)
def test_mixer(feature_extractor, decimal, exception_expectation):
    # Treat it more like a test of "it runs" rather than "it works"
    sr = 8000
    t = np.linspace(0, 1, 8000, dtype=np.float32)
    x1 = np.sin(440.0 * t).reshape(1, -1)
    x2 = np.sin(55.0 * t).reshape(1, -1)

    f1 = feature_extractor.extract(x1, sr)
    f2 = feature_extractor.extract(x2, sr)
    with exception_expectation:
        mixer = FeatureMixer(
            feature_extractor=feature_extractor,
            base_feats=f1,
            frame_shift=feature_extractor.frame_shift,
        )
        mixer.add_to_mix(f2, sampling_rate=sr)

        fmix_feat = mixer.mixed_feats
        fmix_time = feature_extractor.extract(x1 + x2, sr)

        np.testing.assert_almost_equal(fmix_feat, fmix_time, decimal=decimal)

        assert mixer.unmixed_feats.shape == (
            2,
            100,
            feature_extractor.feature_dim(sampling_rate=sr),
        )


def test_feature_mixer_handles_empty_array():
    # Treat it more like a test of "it runs" rather than "it works"
    sr = 16000
    t = np.linspace(0, 1, sr, dtype=np.float32)
    x1 = np.sin(440.0 * t).reshape(1, -1)

    fe = Fbank()
    f1 = fe.extract(x1, sr)
    mixer = FeatureMixer(
        feature_extractor=fe,
        base_feats=f1,
        frame_shift=fe.frame_shift,
    )
    mixer.add_to_mix(np.array([]), sampling_rate=sr)

    fmix_feat = mixer.mixed_feats
    np.testing.assert_equal(fmix_feat, f1)


def test_feature_mixer_handles_empty_array_with_offset():
    # Treat it more like a test of "it runs" rather than "it works"
    sr = 16000
    t = np.linspace(0, 1, sr, dtype=np.float32)
    x1 = np.sin(440.0 * t).reshape(1, -1)

    fe = Fbank()
    f1 = fe.extract(x1, sr)
    mixer = FeatureMixer(
        feature_extractor=fe,
        base_feats=f1,
        frame_shift=fe.frame_shift,
    )
    mixer.add_to_mix(np.array([]), sampling_rate=sr, offset=0.5)

    fmix_feat = mixer.mixed_feats
    # time 0s - 1s: identical values
    np.testing.assert_equal(fmix_feat[:100], f1)
    # time 1s - 1.5s: padding
    np.testing.assert_equal(fmix_feat[100:], -1000)


def test_feature_set_prefix_path():
    features = FeatureSet.from_features(
        [
            Features(
                type="fbank",
                num_frames=1000,
                num_features=40,
                frame_shift=0.01,
                sampling_rate=16000,
                storage_type="lilcom",
                storage_path="feats/",
                storage_key="12345.llc",
                start=0,
                duration=10,
            )
        ]
    )
    for feat in features.with_path_prefix("/data"):
        assert feat.storage_path == "/data/feats"
