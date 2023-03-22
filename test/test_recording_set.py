from functools import lru_cache
from math import isclose

import audioread
import numpy as np
import pytest
from pytest import mark, raises

from lhotse.audio import (
    AudioMixer,
    AudioSource,
    DurationMismatchError,
    Recording,
    RecordingSet,
    set_audio_duration_mismatch_tolerance,
)
from lhotse.testing.dummies import DummyManifest
from lhotse.utils import INT16MAX, fastcopy, is_module_available
from lhotse.utils import nullcontext as does_not_raise


@pytest.fixture
def recording_set() -> RecordingSet:
    return RecordingSet.from_json("test/fixtures/audio.json")


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
    return np.vstack([expected_channel_0(), expected_channel_1()])


@lru_cache(1)
def expected_stereo_single_source() -> np.ndarray:
    """Contents of test/fixtures/stereo.{wav,sph}"""
    return (
        np.vstack(
            [
                np.arange(8000, 16000, dtype=np.int16),
                np.arange(16000, 24000, dtype=np.int16),
            ]
        )
        / INT16MAX
    )


def test_get_metadata(recording_set):
    assert 2 == recording_set.num_channels("recording-1")
    assert 8000 == recording_set.sampling_rate("recording-1")
    assert 4000 == recording_set.num_samples("recording-1")
    assert 0.5 == recording_set.duration("recording-1")


def test_iteration(recording_set):
    assert all(isinstance(item, Recording) for item in recording_set)


def test_get_audio_from_multiple_files(recording_set):
    samples = recording_set.load_audio("recording-1")
    np.testing.assert_almost_equal(samples, expected_stereo_two_sources())


def test_get_stereo_audio_from_single_file(recording_set):
    samples = recording_set.load_audio("recording-2")
    np.testing.assert_almost_equal(samples, expected_stereo_single_source())


def test_load_audio_from_sphere_file(recording_set):
    samples = recording_set.load_audio("recording-3")
    np.testing.assert_almost_equal(samples, expected_stereo_single_source())


@mark.parametrize(
    ["channels", "expected_audio", "exception_expectation"],
    [
        (None, expected_stereo_two_sources(), does_not_raise()),
        (0, expected_channel_0(), does_not_raise()),
        (1, expected_channel_1(), does_not_raise()),
        ([0, 1], expected_stereo_two_sources(), does_not_raise()),
        (1000, "irrelevant", raises(AssertionError)),
    ],
)
def test_get_audio_multichannel(
    recording_set, channels, expected_audio, exception_expectation
):
    with exception_expectation:
        loaded_audio = recording_set.load_audio("recording-1", channels=channels)
        np.testing.assert_almost_equal(loaded_audio, expected_audio)


@mark.parametrize(
    ["duration_tolerance", "exception_expectation"],
    [(0.025, raises(DurationMismatchError)), (0.2, does_not_raise())],
)
def test_get_audio_multichannel_duration_mismatch(
    recording_set, duration_tolerance, exception_expectation
):
    set_audio_duration_mismatch_tolerance(duration_tolerance)
    with exception_expectation:
        recording_set.load_audio("recording-4", channels=[0, 1])


@mark.parametrize(
    [
        "begin_at",
        "duration",
        "expected_start_sample",
        "expected_end_sample",
        "exception_expectation",
    ],
    [
        (0, None, 0, 4000, does_not_raise()),
        (0.1, None, 800, 4000, does_not_raise()),
        (0, 0.3, 0, 2400, does_not_raise()),
        (0.1, 0.2, 800, 2400, does_not_raise()),
        (
            0.3,
            10.0,
            "irrelevant",
            "irrelevant",
            raises(DurationMismatchError),
        ),  # requested more audio than available
    ],
)
def test_get_audio_chunks(
    recording_set,
    begin_at,
    duration,
    expected_start_sample,
    expected_end_sample,
    exception_expectation,
):
    with exception_expectation:
        actual_audio = recording_set.load_audio(
            recording_id="recording-1",
            channels=0,
            offset_seconds=begin_at,
            duration_seconds=duration,
        )
        expected_audio = expected_channel_0()[
            :, expected_start_sample:expected_end_sample
        ]
        np.testing.assert_almost_equal(actual_audio, expected_audio)


def test_add_recording_sets():
    expected = DummyManifest(RecordingSet, begin_id=0, end_id=10)
    recording_set_1 = DummyManifest(RecordingSet, begin_id=0, end_id=5)
    recording_set_2 = DummyManifest(RecordingSet, begin_id=5, end_id=10)
    combined = recording_set_1 + recording_set_2
    assert combined.to_eager() == expected


@pytest.mark.parametrize(
    ["relative_path_depth", "expected_source_path"],
    [
        (None, "test/fixtures/stereo.sph"),
        (1, "stereo.sph"),
        (2, "fixtures/stereo.sph"),
        (3, "test/fixtures/stereo.sph"),
        (4, "test/fixtures/stereo.sph"),
    ],
)
def test_recording_from_sphere(relative_path_depth, expected_source_path):
    rec = Recording.from_file(
        "test/fixtures/stereo.sph", relative_path_depth=relative_path_depth
    )
    assert rec == Recording(
        id="stereo",
        sampling_rate=8000,
        num_samples=8000,
        duration=1.0,
        sources=[
            AudioSource(type="file", channels=[0, 1], source=expected_source_path)
        ],
    )


@pytest.fixture
def file_source():
    return AudioSource(type="file", channels=[0], source="test/fixtures/mono_c0.wav")


@pytest.fixture
def nonfile_source():
    return AudioSource(
        type="command", channels=[0], source="cat test/fixtures/mono_c0.wav"
    )


@pytest.fixture
def recording(file_source):
    return Recording(
        id="rec",
        sources=[file_source, fastcopy(file_source, channels=[1])],
        sampling_rate=8000,
        num_samples=4000,
        duration=0.5,
    )


@pytest.mark.parametrize(
    ["factor", "affix_id"],
    [
        (1.0, True),
        (1.0, False),
        (0.9, True),
        (1.1, True),
    ],
)
def test_recording_perturb_speed(recording, factor, affix_id):
    rec_sp = recording.perturb_speed(factor=factor, affix_id=affix_id)
    if affix_id:
        assert rec_sp.id == f"{recording.id}_sp{factor}"
    else:
        assert rec_sp.id == recording.id
    samples = rec_sp.load_audio()
    assert samples.shape[0] == rec_sp.num_channels
    assert samples.shape[1] == rec_sp.num_samples


@pytest.mark.skipif(
    not is_module_available("nara_wpe"),
    reason="This test requires nara_wpe to be installed.",
)
@pytest.mark.parametrize("affix_id", [True, False])
def test_recording_dereverb_wpe(recording, affix_id):
    rec_wpe = recording.dereverb_wpe(affix_id=affix_id)
    if affix_id:
        assert rec_wpe.id == f"{recording.id}_wpe"
    else:
        assert rec_wpe.id == recording.id
    samples = recording.load_audio()
    samples_wpe = rec_wpe.load_audio()
    assert samples_wpe.shape[0] == rec_wpe.num_channels
    assert samples_wpe.shape[1] == rec_wpe.num_samples
    assert (samples != samples_wpe).any()


@pytest.mark.parametrize(
    ["factor", "affix_id"],
    [
        (1.0, True),
        (1.0, False),
        (0.9, True),
        (1.1, True),
    ],
)
def test_recording_perturb_tempo(recording, factor, affix_id):
    rec_sp = recording.perturb_tempo(factor=factor, affix_id=affix_id)
    if affix_id:
        assert rec_sp.id == f"{recording.id}_tp{factor}"
    else:
        assert rec_sp.id == recording.id
    samples = rec_sp.load_audio()
    assert samples.shape[0] == rec_sp.num_channels
    assert samples.shape[1] == rec_sp.num_samples


@pytest.mark.parametrize(
    ["factor", "affix_id"],
    [
        (1.0, True),
        (1.0, False),
        (0.125, True),
        (0.125, False),
        (2.0, True),
        (2.0, False),
    ],
)
def test_recording_perturb_volume(recording, factor, affix_id):
    rec_vp = recording.perturb_volume(factor=factor, affix_id=affix_id)
    if affix_id:
        assert rec_vp.id == f"{recording.id}_vp{factor}"
    else:
        assert rec_vp.id == recording.id
    samples = rec_vp.load_audio()
    assert samples.shape[0] == rec_vp.num_channels
    assert samples.shape[1] == rec_vp.num_samples


def test_recording_set_perturb_speed(recording_set):
    recs_sp = recording_set.perturb_speed(factor=1.1)
    for r, r_sp in zip(recording_set, recs_sp):
        assert r.duration > r_sp.duration  # Faster recording => shorter duration
        assert r.sampling_rate == r_sp.sampling_rate


def test_recording_set_perturb_tempo(recording_set):
    recs_sp = recording_set.perturb_tempo(factor=1.1)
    for r, r_tp in zip(recording_set, recs_sp):
        assert r.duration > r_tp.duration  # Faster recording => shorter duration
        assert r.sampling_rate == r_tp.sampling_rate


def test_recording_set_perturb_volume(recording_set):
    recs_vp = recording_set.perturb_volume(factor=2.0)
    for r, r_vp in zip(recording_set, recs_vp):
        assert r.duration == r_vp.duration
        assert r.sampling_rate == r_vp.sampling_rate


@pytest.mark.parametrize("sampling_rate", [8000, 16000, 22050, 32000, 44100, 48000])
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
    return RecordingSet.from_recordings(
        [fastcopy(recording, id=f"{recording.id}-{i}") for i in range(5)]
    )


def test_audio_source_path_prefix(file_source):
    assert (
        str(file_source.with_path_prefix("/data").source)
        == "/data/test/fixtures/mono_c0.wav"
    )


def test_audio_source_nonfile_path_prefix(nonfile_source):
    assert (
        str(nonfile_source.with_path_prefix("/data").source)
        == "cat test/fixtures/mono_c0.wav"
    )


def test_recording_path_prefix(recording):
    for source in recording.with_path_prefix("/data").sources:
        assert str(source.source) == "/data/test/fixtures/mono_c0.wav"


def test_recording_set_prefix(recording_set2):
    for recording in recording_set2.with_path_prefix("/data"):
        for source in recording.sources:
            assert str(source.source) == "/data/test/fixtures/mono_c0.wav"


class TestAudioMixer:
    @classmethod
    def setup_class(cls):
        cls.audio1 = np.ones(8000, dtype=np.float32).reshape(1, -1)
        cls.audio2 = np.ones(8000, dtype=np.float32).reshape(1, -1) * 2

    def test_simple_mix(self):
        mixer = AudioMixer(base_audio=self.audio1, sampling_rate=8000)
        mixer.add_to_mix(self.audio2, snr=None, offset=0)

        unmixed = mixer.unmixed_audio
        assert len(unmixed) == 2
        assert all(u.shape == (1, 8000) for u in unmixed)
        assert (unmixed[0] == 1).all()
        assert (unmixed[1] == 2).all()
        assert all(u.dtype == np.float32 for u in unmixed)

        mixed = mixer.mixed_audio
        assert mixed.shape == (1, 8000)
        assert (mixed == 3).all()
        assert mixed.dtype == np.float32

    def test_audio_mixed_with_offset(self):
        mixer = AudioMixer(base_audio=self.audio1, sampling_rate=8000)
        mixer.add_to_mix(self.audio2, snr=None, offset=0.5)

        unmixed = mixer.unmixed_audio
        assert len(unmixed) == 2
        assert all(u.shape == (1, 12000) for u in unmixed)
        assert (unmixed[0][:, :8000] == 1).all()
        assert (unmixed[0][:, 8000:] == 0).all()
        assert (unmixed[1][:, :4000] == 0).all()
        assert (unmixed[1][:, 4000:] == 2).all()
        assert all(u.dtype == np.float32 for u in unmixed)

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
        assert len(unmixed) == 2
        assert all(u.shape == (1, 8000) for u in unmixed)
        assert (unmixed[0] == 1).all()
        np.testing.assert_almost_equal(unmixed[1], 0.31622776)
        assert all(u.dtype == np.float32 for u in unmixed)

        mixed = mixer.mixed_audio
        assert mixed.shape == (1, 8000)
        np.testing.assert_almost_equal(mixed[0, :], 1.31622776)
        assert mixed.dtype == np.float32

    def test_audio_mixed_with_offset_and_snr(self):
        mixer = AudioMixer(base_audio=self.audio1, sampling_rate=8000)
        mixer.add_to_mix(self.audio2, snr=10, offset=0.5)

        unmixed = mixer.unmixed_audio
        assert len(unmixed) == 2
        assert all(u.shape == (1, 12000) for u in unmixed)
        assert (unmixed[0][:, :8000] == 1).all()
        assert (unmixed[0][:, 8000:] == 0).all()
        assert (unmixed[1][:, :4000] == 0).all()
        np.testing.assert_almost_equal(unmixed[1][:, 4000:], 0.31622776)
        assert all(u.dtype == np.float32 for u in unmixed)

        mixed = mixer.mixed_audio
        assert mixed.shape == (1, 12000)
        assert (mixed[0, :4000] == 1).all()
        np.testing.assert_almost_equal(mixed[0, 4000:8000], 1.31622776)
        np.testing.assert_almost_equal(mixed[0, 8000:], 0.31622776)
        assert mixed.dtype == np.float32

    def test_audio_mixer_handles_empty_array(self):
        # Treat it more like a test of "it runs" rather than "it works"
        sr = 16000
        t = np.linspace(0, 1, sr, dtype=np.float32)
        x1 = np.sin(440.0 * t).reshape(1, -1)

        mixer = AudioMixer(
            base_audio=x1,
            sampling_rate=sr,
        )
        mixer.add_to_mix(np.array([]))

        xmix = mixer.mixed_audio
        np.testing.assert_equal(xmix, x1)

    def test_audio_mixer_handles_empty_array_with_offset(self):
        # Treat it more like a test of "it runs" rather than "it works"
        sr = 16000
        t = np.linspace(0, 1, sr, dtype=np.float32)
        x1 = np.sin(440.0 * t).reshape(1, -1)

        mixer = AudioMixer(
            base_audio=x1,
            sampling_rate=sr,
        )
        mixer.add_to_mix(np.array([]), offset=0.5)

        xmix = mixer.mixed_audio
        # 0s - 1s: identical
        np.testing.assert_equal(xmix[:sr], x1)
        # 1s - 1.5s: padding
        np.testing.assert_equal(xmix[sr:], 0)

    def test_audio_mixer_mix_multi_channel_inputs(self):
        sr = 16000
        t = np.linspace(0, 1, sr, dtype=np.float32)
        x1 = np.sin(440.0 * t).reshape(1, -1).repeat(2, axis=0)
        x2 = np.sin(880.0 * t).reshape(1, -1).repeat(2, axis=0)
        y = x1 + x2
        y_mono = y.sum(axis=0, keepdims=True)

        mixer = AudioMixer(
            base_audio=x1,
            sampling_rate=sr,
        )
        mixer.add_to_mix(x2)

        xmix = mixer.mixed_audio
        xmix_mono = mixer.mixed_mono_audio
        np.testing.assert_equal(xmix, y)
        np.testing.assert_equal(xmix_mono, y_mono)

    def test_audio_mixer_mix_mono_with_multi(self):
        sr = 16000
        t = np.linspace(0, 1, sr, dtype=np.float32)
        x1 = np.sin(440.0 * t).reshape(1, -1).repeat(2, axis=0)
        x2 = np.sin(880.0 * t).reshape(1, -1)
        y = x1 + x2
        y_mono = x1.sum(axis=0, keepdims=True) + x2

        mixer = AudioMixer(
            base_audio=x1,
            sampling_rate=sr,
        )
        mixer.add_to_mix(x2)

        xmix = mixer.mixed_audio
        xmix_mono = mixer.mixed_mono_audio
        np.testing.assert_equal(xmix, y)
        np.testing.assert_equal(xmix_mono, y_mono)

    def test_audio_mixer_mix_multi_with_multi_incompatible(self):
        sr = 16000
        t = np.linspace(0, 1, sr, dtype=np.float32)
        x1 = np.sin(440.0 * t).reshape(1, -1).repeat(2, axis=0)
        x2 = np.sin(880.0 * t).reshape(1, -1).repeat(3, axis=0)

        mixer = AudioMixer(
            base_audio=x1,
            sampling_rate=sr,
        )
        with pytest.raises(ValueError):
            mixer.add_to_mix(x2)


@pytest.mark.skipif(
    all(
        "ffmpeg" not in str(backend).lower()
        for backend in audioread.available_backends()
    ),
    reason="Requires FFmpeg to be installed.",
)
def test_opus_recording_from_file():
    path = "test/fixtures/mono_c0.opus"
    recording = Recording.from_file(path)
    # OPUS always overrides the sampling rate to 48000
    assert recording.sampling_rate == 48000
    # OPUS may crate extra audio frames / samples...
    assert isclose(recording.duration, 0.5054166666666666)
    samples = recording.load_audio()
    num_channels, num_samples = samples.shape
    assert num_channels == recording.num_channels
    assert num_samples == recording.num_samples
    assert num_samples == 24260
    # OPUS file read succesfully!


@pytest.mark.skipif(
    all(
        "ffmpeg" not in str(backend).lower()
        for backend in audioread.available_backends()
    ),
    reason="Requires FFmpeg to be installed.",
)
def test_opus_recording_from_file_force_sampling_rate():
    path = "test/fixtures/mono_c0.opus"
    recording = Recording.from_file(path, force_opus_sampling_rate=8000)
    assert recording.sampling_rate == 8000
    assert isclose(recording.duration, 0.5055)
    samples = recording.load_audio()
    num_channels, num_samples = samples.shape
    assert num_channels == recording.num_channels
    assert num_samples == recording.num_samples
    assert num_samples == 4044


@pytest.mark.skipif(
    all(
        "ffmpeg" not in str(backend).lower()
        for backend in audioread.available_backends()
    ),
    reason="Requires FFmpeg to be installed.",
)
def test_opus_stereo_recording_from_file_force_sampling_rate():
    path = "test/fixtures/stereo.opus"
    recording = Recording.from_file(path, force_opus_sampling_rate=8000)
    assert recording.sampling_rate == 8000
    assert isclose(recording.duration, 1.0055)
    samples = recording.load_audio()
    num_channels, num_samples = samples.shape
    assert num_channels == recording.num_channels
    assert num_samples == recording.num_samples
    assert num_samples == 8044


@pytest.mark.skipif(
    all(
        "ffmpeg" not in str(backend).lower()
        for backend in audioread.available_backends()
    ),
    reason="Requires FFmpeg to be installed.",
)
def test_opus_stereo_recording_from_file_force_sampling_rate_read_chunk():
    path = "test/fixtures/stereo.opus"
    recording = Recording.from_file(path, force_opus_sampling_rate=8000)
    assert recording.sampling_rate == 8000
    assert isclose(recording.duration, 1.0055)
    all_samples = recording.load_audio()
    samples = recording.load_audio(offset=0.5, duration=0.25)
    num_channels, num_samples = samples.shape
    assert num_channels == recording.num_channels
    assert num_samples == 2000
    np.testing.assert_almost_equal(samples, all_samples[:, 4000:6000], decimal=5)


def test_audio_source_memory_type(recording):
    memory_recording = recording.move_to_memory()

    np.testing.assert_equal(memory_recording.load_audio(), recording.load_audio())


def test_recording_from_bytes():
    path = "test/fixtures/mono_c0.wav"
    recording = Recording.from_file(path)
    memory_recording = Recording.from_bytes(
        data=open(path, "rb").read(),
        recording_id=recording.id,
    )
    np.testing.assert_equal(memory_recording.load_audio(), recording.load_audio())


def test_memory_recording_dict_serialization():
    path = "test/fixtures/mono_c0.wav"
    rec = Recording.from_bytes(data=open(path, "rb").read(), recording_id="testrec")
    data = rec.to_dict()
    rec_reconstructed = Recording.from_dict(data)
    assert rec == rec_reconstructed
    np.testing.assert_equal(rec_reconstructed.load_audio(), rec.load_audio())
