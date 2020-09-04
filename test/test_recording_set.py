from contextlib import nullcontext as does_not_raise
from functools import lru_cache
from tempfile import NamedTemporaryFile

import numpy as np
import pytest
from pytest import mark, raises

from lhotse.audio import AudioSource, Recording, RecordingSet
from lhotse.test_utils import DummyManifest
from lhotse.utils import INT16MAX


@pytest.fixture
def recording_set() -> RecordingSet:
    return RecordingSet.from_yaml('test/fixtures/audio.yml')


@lru_cache(1)
def expected_channel_0() -> np.ndarray:
    """Contents of test/fixtures/mono_c0.wav"""
    return np.reshape(np.arange(0, 4000) / INT16MAX, (1, -1))


@lru_cache(1)
def expected_channel_1() -> np.ndarray:
    """Contents of test/fixtures/mono_c1.wav"""
    return np.reshape(np.arange(4000, 8000) / INT16MAX, (1, -1))


@lru_cache(1)
def expected_stereo_two_sources() -> np.ndarray:
    """Combined contents of test/fixtures/mono_c{0,1}.wav as channels 0 and 1"""
    return np.vstack([
        expected_channel_0(),
        expected_channel_1()
    ])


@lru_cache(1)
def expected_stereo_single_source() -> np.ndarray:
    """Contents of test/fixtures/stereo.wav"""
    return np.vstack([
        np.arange(8000, 16000, dtype=np.int16),
        np.arange(16000, 24000, dtype=np.int16)
    ]) / INT16MAX


def test_get_metadata(recording_set):
    assert 2 == recording_set.num_channels('recording-1')
    assert 8000 == recording_set.sampling_rate('recording-1')
    assert 4000 == recording_set.num_samples('recording-1')
    assert 0.5 == recording_set.duration_seconds('recording-1')


def test_serialization():
    recording_set = RecordingSet.from_recordings([
        Recording(
            id='x',
            sources=[
                AudioSource(
                    type='file',
                    channel_ids=[0],
                    source='text/fixtures/mono_c0.wav'
                ),
                AudioSource(
                    type='command',
                    channel_ids=[1],
                    source='cat text/fixtures/mono_c1.wav'
                )
            ],
            sampling_rate=8000,
            num_samples=4000,
            duration_seconds=0.5
        )
    ])
    with NamedTemporaryFile() as f:
        recording_set.to_yaml(f.name)
        deserialized = RecordingSet.from_yaml(f.name)
    assert deserialized == recording_set


def test_iteration(recording_set):
    assert all(isinstance(item, Recording) for item in recording_set)


def test_get_audio_from_multiple_files(recording_set):
    samples = recording_set.load_audio('recording-1')
    np.testing.assert_almost_equal(samples, expected_stereo_two_sources())


def test_get_stereo_audio_from_single_file(recording_set):
    samples = recording_set.load_audio('recording-2')
    np.testing.assert_almost_equal(samples, expected_stereo_single_source())


def test_load_audio_from_sphere_file(recording_set):
    samples = recording_set.load_audio('recording-2')
    np.testing.assert_almost_equal(samples, expected_stereo_single_source())


@mark.parametrize(
    ['channels', 'expected_audio', 'exception_expectation'],
    [
        (None, expected_stereo_two_sources(), does_not_raise()),
        (0, expected_channel_0(), does_not_raise()),
        (1, expected_channel_1(), does_not_raise()),
        ([0, 1], expected_stereo_two_sources(), does_not_raise()),
        (1000, 'irrelevant', raises(ValueError))
    ]
)
def test_get_audio_multichannel(recording_set, channels, expected_audio, exception_expectation):
    with exception_expectation:
        loaded_audio = recording_set.load_audio('recording-1', channels=channels)
        np.testing.assert_almost_equal(loaded_audio, expected_audio)


@mark.parametrize(
    ['begin_at', 'duration', 'expected_start_sample', 'expected_end_sample', 'exception_expectation'],
    [
        (None, None, 0, 4000, does_not_raise()),
        (0.1, None, 800, 4000, does_not_raise()),
        (None, 0.3, 0, 2400, does_not_raise()),
        (0.1, 0.2, 800, 2400, does_not_raise()),
        (0.3, 10.0, 'irrelevant', 'irrelevant', raises(ValueError))  # requested more audio than available
    ]
)
def test_get_audio_chunks(
        recording_set,
        begin_at,
        duration,
        expected_start_sample,
        expected_end_sample,
        exception_expectation
):
    with exception_expectation:
        actual_audio = recording_set.load_audio(
            recording_id='recording-1',
            channels=0,
            offset_seconds=begin_at,
            duration_seconds=duration
        )
        expected_audio = expected_channel_0()[:, expected_start_sample: expected_end_sample]
        np.testing.assert_almost_equal(actual_audio, expected_audio)


def test_add_recording_sets():
    expected = DummyManifest(RecordingSet, begin_id=0, end_id=10)
    recording_set_1 = DummyManifest(RecordingSet, begin_id=0, end_id=5)
    recording_set_2 = DummyManifest(RecordingSet, begin_id=5, end_id=10)
    combined = recording_set_1 + recording_set_2
    assert combined == expected


@pytest.mark.parametrize(
    ['relative_path_depth', 'expected_source_path'],
    [
        (None, 'test/fixtures/stereo.sph'),
        (1, 'stereo.sph'),
        (2, 'fixtures/stereo.sph'),
        (3, 'test/fixtures/stereo.sph'),
        (4, 'test/fixtures/stereo.sph')
    ]
)
def test_recording_from_sphere(relative_path_depth, expected_source_path):
    rec = Recording.from_sphere('test/fixtures/stereo.sph', relative_path_depth=relative_path_depth)
    assert rec == Recording(
        id='stereo',
        sampling_rate=8000,
        num_samples=8000,
        duration_seconds=1.0,
        sources=[
            AudioSource(
                type='file',
                channel_ids=[0, 1],
                source=expected_source_path
            )
        ]
    )
