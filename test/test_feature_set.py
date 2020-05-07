from tempfile import NamedTemporaryFile

import numpy as np
import torch
from pytest import mark, param

from lhotse.audio import AudioSet
from lhotse.features import FeatureSet, FeatureExtractor, Features
from lhotse.supervision import SupervisionSet

other_params = {}
some_augmentation = None


@mark.parametrize('feature_type', [
    'mfcc',
    'fbank',
    'spectrogram',
    param('pitch', marks=mark.xfail)
])
def test_feature_extractor(feature_type):
    # Only test that it doesn't crash
    fe = FeatureExtractor(type=feature_type)
    fe.extract(torch.rand(1, 4000), sampling_rate=8000)


def test_feature_extractor_serialization():
    fe = FeatureExtractor()
    with NamedTemporaryFile() as f:
        fe.to_yaml(f.name)
        fe_deserialized = FeatureExtractor.from_yaml(f.name)
    assert fe_deserialized == fe


def test_feature_set_serialization():
    feature_set = FeatureSet(
        feature_extractor=FeatureExtractor(),
        features=[
            Features(
                recording_id='irrelevant',
                channel_id=0,
                start=0.0,
                duration=20.0,
                storage_type='lilcom',
                storage_path='/irrelevant/path.llc'
            )
        ]
    )
    with NamedTemporaryFile() as f:
        feature_set.to_yaml(f.name)
        feature_set_deserialized = FeatureSet.from_yaml(f.name)
    assert feature_set_deserialized == feature_set


def test_create_feature_set():
    # TODO(pzelasko): will be split into several tests

    # Use case #1 - pre-computed features, stored somewhere, specified by an existing manifest
    feature_set = FeatureSet.from_yaml('...')

    # Use case #2
    # - feature extraction
    # - with recordings delivered by AudioSet,
    # - with specified feature parameters (mfcc/fbank/..., frame shapes, etc.),
    # - with specified optional segmentation (default: extract feats for whole recordings)
    # Note: introduce FeatureExtractor as a FeatureSet builder for separation of concerns (loading and manipulating feauters vs extracting them)

    audio_set = AudioSet.from_yaml('...')

    # Variant A: whole recordings
    whole_recording_feature_set: FeatureSet = (
        FeatureExtractor()
            .with_audio_set(audio_set)
            .with_augmentation(some_augmentation)
            .with_algorithm(method='mfcc', frame_size=0.025, frame_shift=0.01, **other_params)
            .extract()
    )

    # Variant B: Exact segments from SupervisionSet
    supervision_set = SupervisionSet.from_yaml('...')

    whole_recording_feature_set: FeatureSet = (
        FeatureExtractor()
            .with_audio_set(audio_set)
            .with_augmentation(some_augmentation)
            .with_algorithm(method='mfcc', frame_size=0.025, frame_shift=0.01, **other_params)
            .with_segmentation(supervision_set, extra_left_seconds=0.0, extra_right_seconds=0.0)
            .extract()
    )

    # Variant C: custom segmentation
    segmentation = [
        #               recording-id  channel  start  duration
        FeatureSegment('recording-1', 0, 0.5, 0.5),
        FeatureSegment('recording-1', 1, 1.3, 1.5),
        FeatureSegment('recording-1', 0, 2.4, 1.1),
        FeatureSegment('recording-2', 0, 0.0, 30.)
    ]
    segment_feature_set: FeatureSet = (
        FeatureExtractor()
            .with_audio_set(audio_set)
            .with_augmentation(some_augmentation)
            .with_algorithm(method='mfcc', frame_size=0.025, frame_shift=0.01, **other_params)
            .with_segmentation(segmentation)
            .extract()
    )


@mark.parametrize(
    ['recording_id', 'channel', 'start', 'duration'],
    [
        ('recording-1', 0, 0.0, None),  # whole recording
        ('recording-2', 0, 1.0, 0.5),
        ('recording-2', 0, 1.5, 0.5),
        ('recording-2', 1, 1.5, 0.5),
        param('recording-nonexistent', 0, 0.0, None, marks=mark.xfail),  # no recording
        param('recording-1', 1000, 0.0, None, marks=mark.xfail),  # no channel
        param('recording-2', 0.0, 0.5, 1.0, marks=mark.xfail),  # no features between [0.0, 1.0]
        param('recording-2', 0.0, 1.5, None, marks=mark.xfail),  # no features after 2.0
    ]
)
def test_load_features(recording_id, channel, start, duration):
    feature_set = FeatureSet([])
    features: np.ndarray = feature_set.load(recording_id, channel, start, duration)
