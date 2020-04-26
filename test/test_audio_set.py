from functools import lru_cache

import numpy as np
from pytest import param, mark

from lhotse.audio import AudioSet, Recording, AudioSource
from lhotse.utils import INT16MAX


@lru_cache(1)
def get_audio_set() -> AudioSet:
    return AudioSet.from_yaml('test/fixtures/audio.yaml')


@lru_cache(1)
def expected_channel_0() -> np.ndarray:
    return np.arange(0, 4000) / INT16MAX


@lru_cache(1)
def expected_channel_1() -> np.ndarray:
    return np.arange(4000, 8000) / INT16MAX


@lru_cache(1)
def expected_stereo() -> np.ndarray:
    return np.vstack([
        expected_channel_0(),
        expected_channel_1()
    ])


def test_get_metadata():
    audio_set = get_audio_set()
    assert 2 == audio_set.num_channels('recording-1')
    assert 8000 == audio_set.sampling_rate('recording-1')
    assert 4000 == audio_set.num_samples('recording-1')
    assert 0.5 == audio_set.duration_seconds('recording-1')


def test_serialization():
    audio_set = AudioSet(recordings={
        'x': Recording(
            id='x',
            sources=[
                AudioSource(
                    type='file',
                    channel_ids=[0],
                    source='text/fixtures/dummy.wav'
                ),
                AudioSource(
                    type='command',
                    channel_ids=[1],
                    source='cat text/fixtures/dummy.wav'
                )
            ],
            sampling_rate=8000,
            num_samples=4000,
            duration_seconds=0.5
        )
    })
    audio_set.to_yaml('.test.yaml')
    deserialized = AudioSet.from_yaml('.test.yaml')
    assert deserialized == audio_set


def test_iteration():
    audio_set = get_audio_set()
    assert all(isinstance(item, Recording) for item in audio_set)


def test_get_audio_from_multiple_files():
    audio_set = get_audio_set()
    samples = audio_set.load_audio('recording-1')
    np.testing.assert_almost_equal(samples, expected_stereo())


@mark.parametrize(
    ['channels', 'expected_audio'],
    [
        (None, expected_stereo()),
        (0, expected_channel_0()),
        (1, expected_channel_1()),
        ([0, 1], expected_stereo()),
        param(1000, 'irrelevant', marks=mark.xfail)
    ]
)
def test_get_audio_multichannel(channels, expected_audio):
    audio_set = get_audio_set()
    np.testing.assert_almost_equal(
        audio_set.load_audio('recording-1', channels=channels),
        expected_audio
    )


@mark.parametrize(
    ['begin_at', 'duration', 'expected_start_sample', 'expected_end_sample'],
    [
        (None, None, 0, 4000),
        (0.1, None, 800, 4000),
        (None, 0.3, 0, 2400),
        (0.1, 0.2, 800, 2400),
        param(0.3, 0.1, 'irrelevant', 'irrelevant', marks=mark.xfail)
    ]
)
def test_get_audio_chunks(begin_at, duration, expected_start_sample, expected_end_sample):
    audio_set = get_audio_set()
    actual_audio = audio_set.load_audio('recording-1', channels=0, offset_seconds=begin_at, duration_seconds=duration)
    expected_audio = expected_channel_0()[expected_start_sample: expected_end_sample]
    np.testing.assert_almost_equal(actual_audio, expected_audio)
