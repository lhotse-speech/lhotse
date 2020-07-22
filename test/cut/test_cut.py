from math import isclose
import pytest

from lhotse.audio import RecordingSet
from lhotse.features import FeatureSet
from lhotse.supervision import SupervisionSet
from lhotse.cut import (
    CutSet,
    make_cuts_from_recordings,
    make_cuts_from_supervisions_features,
    make_cuts_from_supervisions_recordings
)


@pytest.fixture()
def libri_recording_set():
    return RecordingSet.from_yaml('test/fixtures/libri/audio.yml')


@pytest.fixture
def libri_cut_set():
    return CutSet.from_yaml('test/fixtures/libri/cuts.yml')


@pytest.fixture
def libri_features_set():
    return FeatureSet.from_yaml('test/fixtures/libri/feature_manifest.yml')


@pytest.fixture
def supervision_set():
    return SupervisionSet.from_yaml('test/fixtures/supervision.yml')


@pytest.fixture
def libri_cut(libri_cut_set):
    return libri_cut_set['849e13d8-61a2-4d09-a542-dac1aee1b544']


def test_load_audio(libri_cut, libri_recording_set):
    samples = libri_cut.load_audio(libri_recording_set)
    assert samples.shape[0] == 1  # single channel
    assert samples.shape[1] == 10 * 16000  # samples count = duration * sampling_rate


def test_load_audio_by_self(libri_cut):
    # Same as test_load_audio(libri_cut, libri_recording_set), but do not provide external recording_set
    samples = libri_cut.load_audio()
    assert samples.shape == (1, 10 * 16000)


def test_num_frames(libri_cut):
    expected_features_frame_count = round(16.04 / 0.01)  # duration / frame_shift
    assert libri_cut.features.num_frames == expected_features_frame_count

    expected_cut_frame_count = round(10 / 0.01)  # duration / frame_shift
    assert libri_cut.num_frames == expected_cut_frame_count


def test_load_features(libri_cut):
    feats = libri_cut.load_features()
    assert feats.shape[0] == libri_cut.num_frames
    assert feats.shape[1] == libri_cut.features.num_features


def test_load_none_features(libri_cut):
    libri_cut.features = None
    feats = libri_cut.load_features()
    assert feats is None


def test_make_cuts_from_recordings(libri_recording_set):
    expected_duration = 16.04
    cutset = make_cuts_from_recordings(libri_recording_set)
    duration = list(cutset.cuts.values())[0].duration
    assert isclose(duration, expected_duration)


def test_make_cuts_from_supervisions_features(supervision_set, libri_features_set):
    cutset = make_cuts_from_supervisions_features(supervision_set, libri_features_set)
    assert len(cutset) == 2


def test_make_cuts_from_supervisions_recordings(supervision_set, libri_recording_set):
    cutset = make_cuts_from_supervisions_recordings(supervision_set, libri_recording_set)
    assert len(cutset) == 2
