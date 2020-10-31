from contextlib import nullcontext as does_not_raise
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Optional

import numpy as np
import pytest
import torchaudio
from pytest import mark, raises

from lhotse.audio import RecordingSet
from lhotse.augmentation import is_wav_augment_available, WavAugmenter
from lhotse.features import (
    FeatureSet,
    Features,
    FeatureMixer,
    FeatureSetBuilder,
    create_default_feature_extractor,
    Fbank,
    Mfcc,
    Spectrogram, FeatureExtractor
)
from lhotse.features.io import LilcomFilesWriter, LilcomHdf5Writer, NumpyFilesWriter, NumpyHdf5Writer
from lhotse.test_utils import DummyManifest
from lhotse.utils import Seconds, time_diff_to_num_frames

other_params = {}
some_augmentation = None


@mark.parametrize(
    ['feature_type', 'exception_expectation'],
    [
        ('mfcc', does_not_raise()),
        ('fbank', does_not_raise()),
        ('spectrogram', does_not_raise()),
        ('pitch', raises(Exception))
    ]
)
def test_feature_extractor(feature_type, exception_expectation):
    # For now, just test that it runs
    with exception_expectation:
        fe = create_default_feature_extractor(feature_type)
        samples, sr = torchaudio.load('test/fixtures/libri/libri-1088-134315-0000.wav')
        fe.extract(samples=samples, sampling_rate=sr)


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


@pytest.mark.parametrize(
    ['format', 'compressed'],
    [
        ('yaml', False),
        ('yaml', True),
        ('json', False),
        ('json', True),
    ]
)
def test_feature_set_serialization(format, compressed):
    feature_set = FeatureSet(
        features=[
            Features(
                recording_id='irrelevant',
                channels=0,
                start=0.0,
                duration=20.0,
                type='fbank',
                num_frames=2000,
                num_features=20,
                sampling_rate=16000,
                storage_type='lilcom',
                storage_path='/irrelevant/',
                storage_key='path.llc'
            )
        ]
    )
    with NamedTemporaryFile(suffix='.gz' if compressed else '') as f:
        if format == 'json':
            feature_set.to_json(f.name)
            feature_set_deserialized = FeatureSet.from_json(f.name)
        if format == 'yaml':
            feature_set.to_yaml(f.name)
            feature_set_deserialized = FeatureSet.from_yaml(f.name)
    assert feature_set_deserialized == feature_set


@mark.parametrize(
    ['recording_id', 'channel', 'start', 'duration', 'exception_expectation', 'expected_num_frames'],
    [
        ('recording-1', 0, 0.0, None, does_not_raise(), 50),  # whole recording
        ('recording-1', 0, 0.0, 0.499, does_not_raise(), 50),  # practically whole recording
        ('recording-2', 0, 0.0, 0.7, does_not_raise(), 70),
        ('recording-2', 0, 0.5, 0.5, does_not_raise(), 50),
        ('recording-2', 1, 0.25, 0.65, does_not_raise(), 65),
        ('recording-nonexistent', 0, 0.0, None, raises(KeyError), None),  # no recording
        ('recording-1', 1000, 0.0, None, raises(KeyError), None),  # no channel
        ('recording-2', 0, 0.5, 1.0, raises(KeyError), None),  # no features between [1.0, 1.5]
        ('recording-2', 0, 1.5, None, raises(KeyError), None),  # no features after 1.0
    ]
)
def test_load_features(
        recording_id: str,
        channel: int,
        start: float,
        duration: float,
        exception_expectation,
        expected_num_frames: Optional[float]
):
    # just test that it loads
    feature_set = FeatureSet.from_json('test/fixtures/dummy_feats/feature_manifest.json')
    with exception_expectation:
        features = feature_set.load(recording_id, channel_id=channel, start=start, duration=duration)
        # expect a matrix
        assert len(features.shape) == 2
        # expect time as the first dimension
        assert features.shape[0] == expected_num_frames


def test_load_features_with_default_arguments():
    feature_set = FeatureSet.from_json('test/fixtures/dummy_feats/feature_manifest.json')
    features = feature_set.load('recording-1')
    assert features.shape == (50, 23)


@pytest.mark.parametrize(
    'storage', [
        LilcomFilesWriter(TemporaryDirectory().name),
        LilcomHdf5Writer(NamedTemporaryFile().name),
        NumpyFilesWriter(TemporaryDirectory().name),
        NumpyHdf5Writer(NamedTemporaryFile().name)
    ]
)
def test_feature_set_builder(storage):
    recordings: RecordingSet = RecordingSet.from_json('test/fixtures/audio.json')
    extractor = Fbank()
    with storage:
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
        assert features.type == 'fbank'
        # assert that duration is always a multiple of frame_shift
        assert features.num_frames == round(features.duration / features.frame_shift)
        # assert that num_features is preserved
        assert features.num_features == builder.feature_extractor.config.num_mel_bins
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
            sampling_rate=recording.sampling_rate
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


@pytest.mark.skipif(not is_wav_augment_available(), reason='WavAugment required')
def test_feature_set_builder_with_augmentation():
    recordings: RecordingSet = RecordingSet.from_json('test/fixtures/audio.json')
    augmenter = WavAugmenter.create_predefined('pitch_reverb_tdrop', sampling_rate=8000)
    extractor = Fbank()
    with TemporaryDirectory() as d, LilcomFilesWriter(d) as storage:
        builder = FeatureSetBuilder(
            feature_extractor=extractor,
            storage=storage,
            augmenter=augmenter
        )
        feature_set = builder.process_and_store_recordings(recordings=recordings)

        assert len(feature_set) == 6

        feature_infos = list(feature_set)

        # Assert the properties shared by all features
        for features in feature_infos:
            # assert that fbank is the default feature type
            assert features.type == 'fbank'
            # assert that duration is always a multiple of frame_shift
            assert features.num_frames == round(features.duration / features.frame_shift)
            # assert that num_features is preserved
            assert features.num_features == builder.feature_extractor.config.num_mel_bins
            # assert that the storage type metadata matches
            assert features.storage_type == storage.name
            # assert that the metadata is consistent with the data shapes
            arr = features.load()
            assert arr.shape[0] == features.num_frames
            assert arr.shape[1] == features.num_features

        # Assert the properties for recordings of duration 0.5 seconds
        for features in feature_infos[:2]:
            assert features.num_frames == 50
            assert features.duration == 0.5

        # Assert the properties for recordings of duration 1.0 seconds
        for features in feature_infos[2:]:
            assert features.num_frames == 100
            assert features.duration == 1.0


@mark.parametrize(
    ['time_diff', 'frame_length', 'frame_shift', 'expected_num_frames'],
    [
        (1.0, 0.025, 0.01, 98),
        (0.5, 0.025, 0.01, 48),
        (1.0, 0.025, 0.012, 82),
    ]
)
def test_time_diff_to_num_frames(
        time_diff: Seconds,
        frame_length: Seconds,
        frame_shift: Seconds,
        expected_num_frames: int
):
    assert time_diff_to_num_frames(
        time_diff=time_diff, frame_length=frame_length, frame_shift=frame_shift
    ) == expected_num_frames


def test_add_feature_sets():
    expected = DummyManifest(FeatureSet, begin_id=0, end_id=10)
    feature_set_1 = DummyManifest(FeatureSet, begin_id=0, end_id=5)
    feature_set_2 = DummyManifest(FeatureSet, begin_id=5, end_id=10)
    combined = feature_set_1 + feature_set_2
    assert combined == expected


@pytest.mark.parametrize(
    ['feature_extractor', 'decimal', 'exception_expectation'],
    [
        (Fbank(), 0, does_not_raise()),
        (Spectrogram(), -1, does_not_raise()),
        (Mfcc(), None, raises(ValueError)),
    ]
)
def test_mixer(feature_extractor, decimal, exception_expectation):
    # Treat it more like a test of "it runs" rather than "it works"
    t = np.linspace(0, 1, 8000, dtype=np.float32)
    x1 = np.sin(440.0 * t).reshape(1, -1)
    x2 = np.sin(55.0 * t).reshape(1, -1)

    f1 = feature_extractor.extract(x1, 8000)
    f2 = feature_extractor.extract(x2, 8000)
    with exception_expectation:
        mixer = FeatureMixer(
            feature_extractor=feature_extractor,
            base_feats=f1,
            frame_shift=feature_extractor.frame_shift,
        )
        mixer.add_to_mix(f2)

        fmix_feat = mixer.mixed_feats
        fmix_time = feature_extractor.extract(x1 + x2, 8000)

        np.testing.assert_almost_equal(fmix_feat, fmix_time, decimal=decimal)

        assert mixer.unmixed_feats.shape == (2, 100, feature_extractor.feature_dim(sampling_rate=8000))


def test_feature_set_prefix_path():
    features = FeatureSet.from_features([
        Features(
            type='fbank',
            num_frames=1000,
            num_features=40,
            sampling_rate=16000,
            storage_type='lilcom',
            storage_path='feats/',
            storage_key='12345.llc',
            start=0,
            duration=10
        )
    ])
    for feat in features.with_path_prefix('/data'):
        assert feat.storage_path == '/data/feats'
