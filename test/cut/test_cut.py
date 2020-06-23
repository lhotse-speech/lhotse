import pytest

from lhotse.audio import RecordingSet
from lhotse.cut import CutSet


@pytest.fixture()
def libri_recording_set():
    return RecordingSet.from_yaml('test/fixtures/libri/audio.yml')


@pytest.fixture
def libri_cut_set():
    return CutSet.from_yaml('test/fixtures/libri/cuts.yml')


@pytest.fixture
def libri_cut(libri_cut_set):
    return libri_cut_set['849e13d8-61a2-4d09-a542-dac1aee1b544']


def test_load_audio(libri_cut, libri_recording_set):
    samples = libri_cut.load_audio(libri_recording_set)
    assert samples.shape[0] == 1  # single channel
    assert samples.shape[1] == 10 * 16000  # samples count = duration * sampling_rate


def test_num_frames(libri_cut):
    expected_features_frame_count = round(16.04 / 0.01)  # duration / frame_shift
    assert libri_cut.features.num_frames == expected_features_frame_count

    expected_cut_frame_count = round(10 / 0.01)  # duration / frame_shift
    assert libri_cut.num_frames == expected_cut_frame_count


def test_load_features(libri_cut):
    feats = libri_cut.load_features()
    assert feats.shape[0] == libri_cut.num_frames
    assert feats.shape[1] == libri_cut.features.num_features
