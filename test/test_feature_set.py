from math import ceil
from tempfile import NamedTemporaryFile

import torchaudio
from pytest import mark, param

from lhotse.features import FeatureSet, FeatureExtractor, Features, time_diff_to_num_frames
from lhotse.utils import Seconds

other_params = {}
some_augmentation = None


@mark.parametrize('feature_type', [
    'mfcc',
    'fbank',
    'spectrogram',
    param('pitch', marks=mark.xfail)
])
def test_feature_extractor(feature_type):
    # For now, just test that it runs
    # TODO: test that the output is similar to Kaldi
    fe = FeatureExtractor(type=feature_type)
    samples, sr = torchaudio.load('test/fixtures/libri-1088-134315-0000.wav')
    fe.extract(samples=samples, sampling_rate=sr)


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
        ('recording-2', 0, 0.0, 0.7),
        ('recording-2', 0, 0.5, 0.5),
        ('recording-2', 1, 0.25, 0.65),
        param('recording-nonexistent', 0, 0.0, None, marks=mark.xfail),  # no recording
        param('recording-1', 1000, 0.0, None, marks=mark.xfail),  # no channel
        param('recording-2', 0, 0.5, 1.0, marks=mark.xfail),  # no features between [1.0, 1.5]
        param('recording-2', 0, 1.5, None, marks=mark.xfail),  # no features after 1.0
    ]
)
def test_load_features(recording_id: str, channel: int, start: float, duration: float):
    # just test that it loads
    feature_set = FeatureSet.from_yaml('test/fixtures/dummy_feats/feature_manifest.yml')
    features = feature_set.load(recording_id, channel_id=channel, start=start, duration=duration)
    # expect a matrix
    assert len(features.shape) == 2
    # expect time as the first dimension
    fs = feature_set.feature_extractor.spectrogram_config.frame_shift / 1000.0
    if duration is not None:
        # left-hand expression ignores the frame_length - "maximize" the number of frames retained
        # also, allow a lee-way of +/- 2 frames
        assert abs(ceil(duration / fs) - features.shape[0]) <= 2
    # expect frequency as the second dimension
    assert feature_set.feature_extractor.mfcc_config.num_ceps == features.shape[1]


def test_load_features_with_default_arguments():
    feature_set = FeatureSet.from_yaml('test/fixtures/dummy_feats/feature_manifest.yml')
    features = feature_set.load('recording-1')


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
