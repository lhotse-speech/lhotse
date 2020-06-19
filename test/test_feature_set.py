from contextlib import nullcontext as does_not_raise
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import torchaudio
from pytest import mark, raises

from lhotse.audio import AudioSet
from lhotse.features import FeatureSet, FeatureExtractor, Features, FbankMixer, FeatureSetBuilder
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
        ('pitch', raises(ValueError))
    ]
)
def test_feature_extractor(feature_type, exception_expectation):
    # For now, just test that it runs
    # TODO: test that the output is similar to Kaldi
    with exception_expectation:
        fe = FeatureExtractor(type=feature_type)
        samples, sr = torchaudio.load('test/fixtures/libri-1088-134315-0000.wav')
        fe.extract(samples=samples, sampling_rate=sr)


def test_feature_extractor_to_dict_no_defaults():
    fe = FeatureExtractor()
    fe.mfcc_fbank_common_config.num_mel_bins = 80
    fe_dict = fe.to_dict()
    assert fe_dict == {'type': 'fbank', 'mfcc_fbank_common_config': {'num_mel_bins': 80}}


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
                type='fbank',
                num_frames=2000,
                num_features=20,
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
    ['recording_id', 'channel', 'start', 'duration', 'exception_expectation'],
    [
        ('recording-1', 0, 0.0, None, does_not_raise()),  # whole recording
        ('recording-2', 0, 0.0, 0.7, does_not_raise()),
        ('recording-2', 0, 0.5, 0.5, does_not_raise()),
        ('recording-2', 1, 0.25, 0.65, does_not_raise()),
        ('recording-nonexistent', 0, 0.0, None, raises(KeyError)),  # no recording
        ('recording-1', 1000, 0.0, None, raises(KeyError)),  # no channel
        ('recording-2', 0, 0.5, 1.0, raises(KeyError)),  # no features between [1.0, 1.5]
        ('recording-2', 0, 1.5, None, raises(KeyError)),  # no features after 1.0
    ]
)
def test_load_features(recording_id: str, channel: int, start: float, duration: float, exception_expectation):
    # just test that it loads
    feature_set = FeatureSet.from_yaml('test/fixtures/dummy_feats/feature_manifest.yml')
    with exception_expectation:
        features = feature_set.load(recording_id, channel_id=channel, start=start, duration=duration)
        # expect a matrix
        assert len(features.shape) == 2
        # expect time as the first dimension
        frame_shift = feature_set.feature_extractor.spectrogram_config.frame_shift
        if duration is not None:
            # left-hand expression ignores the frame_length - "maximize" the number of frames retained
            # also, allow a lee-way of +/- 2 frames
            assert duration / frame_shift == features.shape[0]
        # expect frequency as the second dimension
        assert feature_set.feature_extractor.mfcc_fbank_common_config.num_mel_bins == features.shape[1]


def test_load_features_with_default_arguments():
    feature_set = FeatureSet.from_yaml('test/fixtures/dummy_feats/feature_manifest.yml')
    features = feature_set.load('recording-1')


def test_feature_set_builder():
    audio_set = AudioSet.from_yaml('test/fixtures/audio.yml')
    with TemporaryDirectory() as output_dir:
        builder = FeatureSetBuilder(feature_extractor=FeatureExtractor(), output_dir=output_dir)
        feature_set = builder.process_and_store_recordings(recordings=audio_set)

    assert len(feature_set) == 4

    feature_infos = list(feature_set)

    # Assert the properties shared by all features
    for features in feature_infos:
        # assert that fbank is the default feature type
        assert features.type == 'fbank'
        # assert that duration is always a multiple of frame_shift
        assert features.num_frames == round(features.duration / features.frame_shift)
        # assert that num_features is preserved
        assert features.num_features == builder.feature_extractor.mfcc_fbank_common_config.num_mel_bins
        # assert that lilcom is the default storate type
        assert features.storage_type == 'lilcom'

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


def test_overlay_fbank():
    # Treat it more like a test of "it runs" rather than "it works"
    t = np.linspace(0, 1, 8000, dtype=np.float32)
    x1 = np.sin(440.0 * t).reshape(1, -1)
    x2 = np.sin(55.0 * t).reshape(1, -1)

    feature_extractor = FeatureExtractor(type='fbank')
    f1 = feature_extractor.extract(x1, 8000).numpy()
    f2 = feature_extractor.extract(x2, 8000).numpy()
    mixer = FbankMixer(
        base_feats=f1,
        frame_shift=feature_extractor.spectrogram_config.frame_shift,
    )
    mixer.add_to_mix(f2)

    fmix_feat = mixer.mixed_feats
    fmix_time = feature_extractor.extract(x1 + x2, 8000).numpy()

    np.testing.assert_almost_equal(fmix_feat, fmix_time, decimal=0)
