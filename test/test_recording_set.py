from functools import lru_cache

import audioread
import numpy as np
import pytest
from pytest import mark, raises

from lhotse.audio import AudioMixer, AudioSource, Recording, RecordingSet
from lhotse.testing.dummies import DummyManifest
from lhotse.utils import INT16MAX
from lhotse.utils import fastcopy, nullcontext as does_not_raise


@pytest.fixture
def recording_set() -> RecordingSet:
    return RecordingSet.from_json('test/fixtures/audio.json')


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
    """Contents of test/fixtures/stereo.{wav,sph}"""
    return np.vstack([
        np.arange(8000, 16000, dtype=np.int16),
        np.arange(16000, 24000, dtype=np.int16)
    ]) / INT16MAX


def test_get_metadata(recording_set):
    assert 2 == recording_set.num_channels('recording-1')
    assert 8000 == recording_set.sampling_rate('recording-1')
    assert 4000 == recording_set.num_samples('recording-1')
    assert 0.5 == recording_set.duration('recording-1')


def test_iteration(recording_set):
    assert all(isinstance(item, Recording) for item in recording_set)


def test_get_audio_from_multiple_files(recording_set):
    samples = recording_set.load_audio('recording-1')
    np.testing.assert_almost_equal(samples, expected_stereo_two_sources())


def test_get_stereo_audio_from_single_file(recording_set):
    samples = recording_set.load_audio('recording-2')
    np.testing.assert_almost_equal(samples, expected_stereo_single_source())


def test_load_audio_from_sphere_file(recording_set):
    samples = recording_set.load_audio('recording-3')
    np.testing.assert_almost_equal(samples, expected_stereo_single_source())


@mark.parametrize(
    ['channels', 'expected_audio', 'exception_expectation'],
    [
        (None, expected_stereo_two_sources(), does_not_raise()),
        (0, expected_channel_0(), does_not_raise()),
        (1, expected_channel_1(), does_not_raise()),
        ([0, 1], expected_stereo_two_sources(), does_not_raise()),
        (1000, 'irrelevant', raises(AssertionError))
    ]
)
def test_get_audio_multichannel(recording_set, channels, expected_audio, exception_expectation):
    with exception_expectation:
        loaded_audio = recording_set.load_audio('recording-1', channels=channels)
        np.testing.assert_almost_equal(loaded_audio, expected_audio)


@mark.parametrize(
    ['begin_at', 'duration', 'expected_start_sample', 'expected_end_sample', 'exception_expectation'],
    [
        (0, None, 0, 4000, does_not_raise()),
        (0.1, None, 800, 4000, does_not_raise()),
        (0, 0.3, 0, 2400, does_not_raise()),
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
    rec = Recording.from_file('test/fixtures/stereo.sph', relative_path_depth=relative_path_depth)
    assert rec == Recording(
        id='stereo',
        sampling_rate=8000,
        num_samples=8000,
        duration=1.0,
        sources=[
            AudioSource(
                type='file',
                channels=[0, 1],
                source=expected_source_path
            )
        ]
    )


@pytest.fixture
def file_source():
    return AudioSource(type='file', channels=[0], source='test/fixtures/mono_c0.wav')


@pytest.fixture
def nonfile_source():
    return AudioSource(type='command', channels=[0], source='cat test/fixtures/mono_c0.wav')


@pytest.fixture
def recording(file_source):
    return Recording(id='rec', sources=[file_source, fastcopy(file_source, channels=[1])], sampling_rate=8000,
                     num_samples=4000, duration=0.5)


@pytest.mark.parametrize(
    ['factor', 'affix_id'],
    [
        (1.0, True),
        (1.0, False),
        (0.9, True),
        (1.1, True),
    ]
)
def test_recording_perturb_speed(recording, factor, affix_id):
    rec_sp = recording.perturb_speed(factor=factor, affix_id=affix_id)
    if affix_id:
        assert rec_sp.id == f'{recording.id}_sp{factor}'
    else:
        assert rec_sp.id == recording.id
    samples = rec_sp.load_audio()
    assert samples.shape[0] == rec_sp.num_channels
    assert samples.shape[1] == rec_sp.num_samples


def test_recording_set_perturb_speed(recording_set):
    recs_sp = recording_set.perturb_speed(factor=1.1)
    for r, r_sp in zip(recording_set, recs_sp):
        assert r.duration > r_sp.duration  # Faster recording => shorter duration
        assert r.sampling_rate == r_sp.sampling_rate


@pytest.mark.parametrize('sampling_rate', [8000, 16000, 22050, 32000, 44100, 48000])
def test_recording_resample(recording, sampling_rate):
    rec_sp = recording.resample(sampling_rate)
    assert rec_sp.id == recording.id
    assert rec_sp.duration == recording.duration
    samples = rec_sp.load_audio()
    assert samples.shape[0] == rec_sp.num_channels
    assert samples.shape[1] == rec_sp.num_samples


def test_recording_set_resample(recording_set):
    recs_sp = recording_set.resample(sampling_rate=44100)
    for r, r_sp in zip(recording_set, recs_sp):
        assert r.duration == r_sp.duration
        assert r_sp.sampling_rate == 44100
        assert r_sp.num_samples > r.num_samples


@pytest.fixture
def recording_set2(recording):
    return RecordingSet.from_recordings([
        fastcopy(recording, id=f'{recording.id}-{i}') for i in range(5)
    ])


def test_audio_source_path_prefix(file_source):
    assert str(file_source.with_path_prefix('/data').source) == '/data/test/fixtures/mono_c0.wav'


def test_audio_source_nonfile_path_prefix(nonfile_source):
    assert str(nonfile_source.with_path_prefix('/data').source) == 'cat test/fixtures/mono_c0.wav'


def test_recording_path_prefix(recording):
    for source in recording.with_path_prefix('/data').sources:
        assert str(source.source) == '/data/test/fixtures/mono_c0.wav'


def test_recording_set_prefix(recording_set2):
    for recording in recording_set2.with_path_prefix('/data'):
        for source in recording.sources:
            assert str(source.source) == '/data/test/fixtures/mono_c0.wav'


class TestAudioMixer:
    @classmethod
    def setup_class(cls):
        cls.audio1 = np.ones(8000, dtype=np.float32).reshape(1, -1)
        cls.audio2 = np.ones(8000, dtype=np.float32).reshape(1, -1) * 2

    def test_simple_mix(self):
        mixer = AudioMixer(base_audio=self.audio1, sampling_rate=8000)
        mixer.add_to_mix(self.audio2, snr=None, offset=0)

        unmixed = mixer.unmixed_audio
        assert unmixed.shape == (2, 8000)
        assert (unmixed[0, :] == 1).all()
        assert (unmixed[1, :] == 2).all()
        assert unmixed.dtype == np.float32

        mixed = mixer.mixed_audio
        assert mixed.shape == (1, 8000)
        assert (mixed == 3).all()
        assert mixed.dtype == np.float32

    def test_audio_mixed_with_offset(self):
        mixer = AudioMixer(base_audio=self.audio1, sampling_rate=8000)
        mixer.add_to_mix(self.audio2, snr=None, offset=0.5)

        unmixed = mixer.unmixed_audio
        assert unmixed.shape == (2, 12000)  # offset 0.5s == 4000 samples
        assert (unmixed[0, :8000] == 1).all()
        assert (unmixed[0, 8000:] == 0).all()
        assert (unmixed[1, :4000] == 0).all()
        assert (unmixed[1, 4000:] == 2).all()
        assert unmixed.dtype == np.float32

        mixed = mixer.mixed_audio
        assert mixed.shape == (1, 12000)
        assert (mixed[0, :4000] == 1).all()
        assert (mixed[0, 4000:8000] == 3).all()
        assert (mixed[0, 8000:] == 2).all()
        assert mixed.dtype == np.float32

    def test_audio_mixed_with_snr(self):
        mixer = AudioMixer(base_audio=self.audio1, sampling_rate=8000)
        mixer.add_to_mix(self.audio2, snr=10, offset=0)

        unmixed = mixer.unmixed_audio
        assert unmixed.shape == (2, 8000)
        assert (unmixed[0, :] == 1).all()
        np.testing.assert_almost_equal(unmixed[1, :], 0.31622776)
        assert unmixed.dtype == np.float32

        mixed = mixer.mixed_audio
        assert mixed.shape == (1, 8000)
        np.testing.assert_almost_equal(mixed[0, :], 1.31622776)
        assert mixed.dtype == np.float32

    def test_audio_mixed_with_offset_and_snr(self):
        mixer = AudioMixer(base_audio=self.audio1, sampling_rate=8000)
        mixer.add_to_mix(self.audio2, snr=10, offset=0.5)

        unmixed = mixer.unmixed_audio
        assert unmixed.shape == (2, 12000)  # offset 0.5s == 4000 samples
        assert (unmixed[0, :8000] == 1).all()
        assert (unmixed[0, 8000:] == 0).all()
        assert (unmixed[1, :4000] == 0).all()
        np.testing.assert_almost_equal(unmixed[1, 4000:], 0.31622776)
        assert unmixed.dtype == np.float32

        mixed = mixer.mixed_audio
        assert mixed.shape == (1, 12000)
        assert (mixed[0, :4000] == 1).all()
        np.testing.assert_almost_equal(mixed[0, 4000:8000], 1.31622776)
        np.testing.assert_almost_equal(mixed[0, 8000:], 0.31622776)
        assert mixed.dtype == np.float32


@pytest.mark.skipif(
    all('ffmpeg' not in str(backend).lower() for backend in audioread.available_backends()),
    reason='Requires FFmpeg to be installed.'
)
def test_recording_from_file_using_audioread():
    path = 'test/fixtures/mono_c0.opus'
    recording = Recording.from_file(path)
    recording.load_audio()
    # OPUS file read succesfully!
