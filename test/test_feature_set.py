from tempfile import NamedTemporaryFile

import numpy as np
import torch
from pytest import mark, param

from lhotse.features import FeatureSet, FeatureExtractor, Features

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
    feature_set = FeatureSet(feature_extractor=FeatureExtractor())
    features: np.ndarray = feature_set.load(recording_id, channel, start, duration)
