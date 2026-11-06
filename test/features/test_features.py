import pickle
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import List, Optional

import numpy as np
import pytest
from pytest import mark, raises

pytest.importorskip("lilcom", reason="Lilcom tests require lilcom.")

from test.markers import requires_sph2pipe

from lhotse import FbankConfig, MfccConfig
from lhotse.audio import RecordingSet
from lhotse.features import (
    Fbank,
    FeatureMixer,
    Features,
    KaldifeatFbank,
    LibrosaFbank,
    LogSpectrogram,
    Mfcc,
    Spectrogram,
    TorchaudioFbank,
    TorchaudioSpectrogram,
    TorchaudioSpectrogramConfig,
    WhisperFbank,
)
from lhotse.features.base import compute_global_stats
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
from lhotse.serialization import load_json
from lhotse.utils import Seconds, is_module_available
from lhotse.utils import nullcontext as does_not_raise
from lhotse.utils import time_diff_to_num_frames

other_params = {}
some_augmentation = None


LEGACY_FEATURES_MANIFEST = "test/fixtures/dummy_feats/feature_manifest.json"


def load_legacy_features() -> List[Features]:
    return [Features.from_dict(d) for d in load_json(LEGACY_FEATURES_MANIFEST)]


@mark.parametrize(
    ["index", "start", "duration", "expected_num_frames"],
    [
        (0, 0.0, None, 50),  # whole recording
        (0, 0.0, 0.499, 50),  # practically whole recording
        (2, 0.0, 0.7, 70),
        (2, 0.5, 0.5, 50),
        (3, 0.25, 0.65, 65),
    ],
)
def test_load_features(
    index: int,
    start: float,
    duration: Optional[float],
    expected_num_frames: int,
):
    features = load_legacy_features()[index].load(start=start, duration=duration)
    assert features.shape == (expected_num_frames, 23)
    assert features.dtype == np.float32


def test_load_features_with_default_arguments():
    features = load_legacy_features()[0].load()
    assert features.shape == (50, 23)


def test_compute_global_stats():
    features = load_legacy_features()
    with NamedTemporaryFile() as f:
        stats = compute_global_stats(features, storage_path=f.name)
        f.flush()
        read_stats = pickle.load(f)
    # Post-condition 1: feature dim is consistent
    assert stats["norm_means"].shape == (features[0].num_features,)
    assert stats["norm_stds"].shape == (features[0].num_features,)
    # Post-condition 2: the iterative method yields very close results to
    # the "standard" method.
    true_means = np.mean(np.concatenate([f.load() for f in features]), axis=0)
    true_stds = np.std(np.concatenate([f.load() for f in features]), axis=0)
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
@requires_sph2pipe
def test_extract_from_recording_and_store(storage_fn):
    recordings: RecordingSet = RecordingSet.from_json(
        "test/fixtures/audio.json"
    ).filter(lambda r: r.id != "recording-4")
    extractor = Fbank(FbankConfig(sampling_rate=8000))
    with storage_fn() as storage:
        feature_infos = [
            extractor.extract_from_recording_and_store(
                recording=recording, storage=storage, channels=channel
            )
            for recording in recordings
            for channel in recording.channel_ids
        ]

    assert len(feature_infos) == 6

    # Assert the properties shared by all features
    for features in feature_infos:
        # assert that fbank is the default feature type
        assert features.type == "kaldi-fbank"
        # assert that duration is always a multiple of frame_shift
        assert features.num_frames == round(features.duration / features.frame_shift)
        # assert that num_features is preserved
        assert features.num_features == extractor.config.num_filters
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


@pytest.mark.parametrize(
    ["feature_extractor", "decimal", "exception_expectation"],
    [
        (Fbank(FbankConfig(num_filters=40, sampling_rate=8000)), 0, does_not_raise()),
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


@pytest.mark.parametrize(
    "feature_extractor_type",
    [
        Fbank,
        Spectrogram,
        LogSpectrogram,
        TorchaudioFbank,
        TorchaudioSpectrogram,
        LibrosaFbank,
        WhisperFbank,
        KaldifeatFbank,
    ],
)
def test_compute_energy_does_not_depend_on_num_frames(feature_extractor_type):
    feats = np.random.default_rng(0).uniform(0.1, 1.0, size=(20, 40))
    repeated = np.tile(feats, (5, 1))
    assert feature_extractor_type.compute_energy(repeated) == pytest.approx(
        feature_extractor_type.compute_energy(feats)
    )


def test_feature_mixer_snr_does_not_depend_on_noise_length():
    sr = 16000
    t = np.linspace(0, 1, sr, dtype=np.float32)
    speech = np.sin(440.0 * t).reshape(1, -1)
    noise = 0.1 * np.sin(55.0 * t[: sr // 5]).reshape(1, -1)  # 0.2s

    fe = Fbank()
    speech_feats = fe.extract(speech, sr)
    noise_feats = fe.extract(noise, sr)

    # Repeating the noise 5 times does not change its level, so the gain must not change
    gains = []
    for feats in [noise_feats, np.tile(noise_feats, (5, 1))]:
        mixer = FeatureMixer(
            feature_extractor=fe,
            base_feats=speech_feats,
            frame_shift=fe.frame_shift,
        )
        mixer.add_to_mix(feats, sampling_rate=sr, snr=10)
        gains.append(mixer.gains[-1])

    expected = (
        10 ** (-10 / 10) * np.mean(np.exp(speech_feats)) / np.mean(np.exp(noise_feats))
    )
    assert gains[0] == pytest.approx(expected, rel=1e-5)
    assert gains[1] == pytest.approx(expected, rel=1e-5)


def test_features_with_path_prefix():
    features = Features(
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
    assert features.with_path_prefix("/data").storage_path == "/data/feats"
    assert features.storage_path == "feats/"
