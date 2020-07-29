from contextlib import nullcontext as does_not_raise
from functools import lru_cache
from tempfile import NamedTemporaryFile

import numpy as np
from pytest import mark, raises

from lhotse.audio import AudioSource, Recording, RecordingSet
from lhotse.test_utils import DummyManifest
from lhotse.utils import INT16MAX


@lru_cache(1)
def get_audio_set() -> RecordingSet:
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


def test_get_metadata():
    audio_set = get_audio_set()
    assert 2 == audio_set.num_channels('recording-1')
    assert 8000 == audio_set.sampling_rate('recording-1')
    assert 4000 == audio_set.num_samples('recording-1')
    assert 0.5 == audio_set.duration_seconds('recording-1')


def test_serialization():
    audio_set = RecordingSet.from_recordings([
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
        audio_set.to_yaml(f.name)
        deserialized = RecordingSet.from_yaml(f.name)
    assert deserialized == audio_set


def test_iteration():
    audio_set = get_audio_set()
    assert all(isinstance(item, Recording) for item in audio_set)


def test_get_audio_from_multiple_files():
    audio_set = get_audio_set()
    samples = audio_set.load_audio('recording-1')
    np.testing.assert_almost_equal(samples, expected_stereo_two_sources())


def test_get_stereo_audio_from_single_file():
    audio_set = get_audio_set()
    samples = audio_set.load_audio('recording-2')
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
def test_get_audio_multichannel(channels, expected_audio, exception_expectation):
    audio_set = get_audio_set()
    with exception_expectation:
        loaded_audio = audio_set.load_audio('recording-1', channels=channels)
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
def test_get_audio_chunks(begin_at, duration, expected_start_sample, expected_end_sample, exception_expectation):
    audio_set = get_audio_set()
    with exception_expectation:
        actual_audio = audio_set.load_audio(
            recording_id='recording-1',
            channels=0,
            offset_seconds=begin_at,
            duration_seconds=duration
        )
        expected_audio = expected_channel_0()[:, expected_start_sample: expected_end_sample]
        np.testing.assert_almost_equal(actual_audio, expected_audio)


def test_add_audio_sets():
    expected = DummyManifest(RecordingSet, begin_id=0, end_id=10)
    audio_set_1 = DummyManifest(RecordingSet, begin_id=0, end_id=5)
    audio_set_2 = DummyManifest(RecordingSet, begin_id=5, end_id=10)
    combined = audio_set_1 + audio_set_2
    assert combined == expected
