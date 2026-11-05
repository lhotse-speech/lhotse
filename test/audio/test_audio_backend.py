from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

import lhotse
from lhotse import Recording
from lhotse.audio import AudioLoadingError, VideoInfo
from lhotse.audio.backend import (
    CompositeAudioBackend,
    LibsndfileBackend,
    TorchaudioDefaultBackend,
    TorchaudioFFMPEGBackend,
    TorchcodecBackend,
    check_torchaudio_version_gt,
    torchaudio_ffmpeg_backend_available,
    torchaudio_soundfile_supports_format,
)
from lhotse.testing.random import deterministic_rng
from lhotse.utils import INT16MAX, is_torchcodec_available


def test_default_audio_backend():
    lhotse.audio.backend.CURRENT_AUDIO_BACKEND = None
    b = lhotse.get_current_audio_backend()
    assert isinstance(b, CompositeAudioBackend)


def test_list_available_audio_backends():
    assert lhotse.available_audio_backends() == [
        "default",
        "AudioreadBackend",
        "CompositeAudioBackend",
        "FfmpegSubprocessOpusBackend",
        "FfmpegTorchaudioStreamerBackend",
        "LibsndfileBackend",
        "Sph2pipeSubprocessBackend",
        "TorchaudioDefaultBackend",
        "TorchaudioFFMPEGBackend",
        "TorchcodecBackend",
    ]


@pytest.mark.parametrize("backend", ["LibsndfileBackend", LibsndfileBackend()])
def test_audio_backend_contextmanager(backend):
    lhotse.audio.backend.CURRENT_AUDIO_BACKEND = None
    assert isinstance(lhotse.get_current_audio_backend(), CompositeAudioBackend)
    with lhotse.audio_backend(backend):
        assert isinstance(lhotse.get_current_audio_backend(), LibsndfileBackend)
    assert isinstance(lhotse.get_current_audio_backend(), CompositeAudioBackend)


@pytest.fixture()
def backend_set_via_env_var(monkeypatch):
    lhotse.audio.backend.CURRENT_AUDIO_BACKEND = None
    monkeypatch.setenv("LHOTSE_AUDIO_BACKEND", "LibsndfileBackend")
    yield
    lhotse.set_current_audio_backend("default")


def test_envvar_audio_backend(backend_set_via_env_var):
    b = lhotse.get_current_audio_backend()
    assert isinstance(b, LibsndfileBackend)


@pytest.mark.skipif(
    not torchaudio_soundfile_supports_format(), reason="Requires torchaudio v0.9.0+"
)
@pytest.mark.parametrize("backend", lhotse.available_audio_backends())
@pytest.mark.parametrize("format", ["wav", "flac", "opus"])
def test_save_and_load(deterministic_rng, tmp_path, backend, format):

    if (
        "Torchaudio" in backend
        and format == "opus"
        and not check_torchaudio_version_gt("2.1.0")
    ):
        return
    if backend == "CompositeAudioBackend":
        return

    path = tmp_path / f"test.{format}"

    with lhotse.audio_backend(backend) as backend_inst:
        if not backend_inst.supports_save():
            return
        if not backend_inst.is_applicable(path):
            return

        audio = (np.random.randint(0, INT16MAX, size=(1, 16000)) / INT16MAX).astype(
            np.float32
        )
        lhotse.audio.backend.save_audio(path, audio, sampling_rate=16000, format=format)
        restored, sr = lhotse.audio.backend.read_audio(path)

        if format != "opus":
            np.testing.assert_allclose(audio, restored)
        else:
            if backend in ("LibsndfileBackend", "default"):
                # only libnsdfile auto-resamples OPUS
                assert restored.shape == audio.shape
            else:
                assert restored.shape == (1, 48000)


@pytest.mark.skipif(
    not check_torchaudio_version_gt("2.1.0"),
    reason="Older torchaudio versions don't support most configurations in this test.",
)
@pytest.mark.parametrize(
    ["backend_save", "backend_read"],
    [
        pytest.param(
            LibsndfileBackend,
            TorchaudioFFMPEGBackend,
            marks=[
                pytest.mark.skipif(
                    not torchaudio_ffmpeg_backend_available(),
                    reason="Requires Torchaudio + FFMPEG",
                ),
            ],
        ),
        pytest.param(LibsndfileBackend, TorchaudioDefaultBackend),
        pytest.param(TorchaudioDefaultBackend, LibsndfileBackend),
        pytest.param(
            TorchaudioDefaultBackend,
            TorchaudioFFMPEGBackend,
            marks=[
                pytest.mark.skipif(
                    not torchaudio_ffmpeg_backend_available(),
                    reason="Requires Torchaudio + FFMPEG",
                ),
            ],
        ),
        pytest.param(
            TorchaudioFFMPEGBackend,
            LibsndfileBackend,
            marks=pytest.mark.skipif(
                not torchaudio_ffmpeg_backend_available(),
                reason="Requires Torchaudio + FFMPEG",
            ),
        ),
        pytest.param(
            TorchaudioFFMPEGBackend,
            TorchaudioDefaultBackend,
            marks=pytest.mark.skipif(
                not torchaudio_ffmpeg_backend_available(),
                reason="Requires Torchaudio + FFMPEG",
            ),
        ),
    ],
)
def test_save_load_opus_different_backends(
    deterministic_rng, tmp_path, backend_save, backend_read
):
    with lhotse.audio_backend(backend_save):
        audio = (np.random.randint(0, INT16MAX, size=(1, 16000)) / INT16MAX).astype(
            np.float32
        )
        path = tmp_path / "test.opus"
        lhotse.audio.backend.save_audio(path, audio, sampling_rate=16000, format="opus")

        recording_old = lhotse.Recording.from_file(path)

    with lhotse.audio_backend(backend_read):
        # Raw read/save utilities work across backends
        restored, sr = lhotse.audio.backend.read_audio(path)
        if backend_read == LibsndfileBackend:
            assert restored.shape == (1, 16000)
        else:
            assert restored.shape == (1, 48000)

        # lhotse Recording doesn't work when it was created with backend1 and is read with incompatible backend2
        if backend_save == LibsndfileBackend or backend_read == LibsndfileBackend:
            with pytest.raises(AudioLoadingError):
                restored2 = recording_old.load_audio()
        else:
            # but works otherwise (e.g. using different torchaudio versions)
            restored2 = recording_old.load_audio()
            np.testing.assert_allclose(restored2, restored)
            recording_old.to_cut().truncate(duration=0.2).move_to_memory(
                audio_format="wav"
            ).load_audio()

        # lhotse Recording works when created with the same backend but data saved using different backend
        recording_new = lhotse.Recording.from_file(path)
        restored3 = recording_new.load_audio()
        np.testing.assert_allclose(restored3, restored)
        # transcoding does not raise exception
        recording_new.to_cut().truncate(duration=0.2).move_to_memory(
            audio_format="wav"
        ).load_audio()


@pytest.mark.parametrize("backend", ["default", LibsndfileBackend])
def test_audio_info_from_bytes_io(backend):
    audio_filelike = BytesIO(open("test/fixtures/mono_c0.wav", "rb").read())
    with lhotse.audio_backend(backend):
        meta = lhotse.audio.info(audio_filelike)
        assert meta.duration == 0.5
        assert meta.frames == 4000
        assert meta.samplerate == 8000
        assert meta.channels == 1


@pytest.mark.skipif(not is_torchcodec_available(), reason="Requires torchcodec")
@pytest.mark.parametrize(
    "path",
    [
        "test/fixtures/mono_c0.wav",
        "test/fixtures/stereo.wav",
        "test/fixtures/mono_c0.opus",
        "test/fixtures/stereo.opus",
        "test/fixtures/common_voice_en_651325.mp3",
    ],
)
def test_torchcodec_info_and_read(path):
    backend = TorchcodecBackend()

    meta = backend.info(path)
    assert meta.samplerate > 0
    assert meta.channels > 0
    assert meta.duration > 0

    audio, sr = backend.read_audio(path)
    assert sr == meta.samplerate
    assert audio.shape[0] == meta.channels
    assert audio.dtype == np.float32


@pytest.mark.skipif(not is_torchcodec_available(), reason="Requires torchcodec")
@pytest.mark.parametrize(
    "path", ["test/fixtures/mono_c0.wav", "test/fixtures/stereo.wav"]
)
@pytest.mark.parametrize("offset", [0, 0.1])
@pytest.mark.parametrize("duration", [None, 0.1])
def test_torchcodec_read_with_offset_duration(path, offset, duration):
    backend = TorchcodecBackend()
    audio, sr = backend.read_audio(path, offset=offset, duration=duration)
    assert audio.ndim == 2
    if duration is not None:
        expected_samples = int(round(duration * sr))
        # Allow 1 sample tolerance for rounding
        assert abs(audio.shape[1] - expected_samples) <= 1


@pytest.mark.skipif(not is_torchcodec_available(), reason="Requires torchcodec")
@pytest.mark.parametrize("offset", [0.0, 0.3])
def test_torchcodec_read_mp3_with_offset_matches_full_read(offset):
    # The first decoded sample of this MP3 is timestamped 0.138 s (encoder delay).
    backend = TorchcodecBackend()
    full, sr = backend.read_audio("test/fixtures/stereo.mp3")
    audio, _ = backend.read_audio(
        "test/fixtures/stereo.mp3", offset=offset, duration=0.5
    )
    start = round(offset * sr)
    assert audio.shape == (2, round(0.5 * sr))
    np.testing.assert_array_equal(audio, full[:, start : start + audio.shape[1]])


@pytest.mark.skipif(not is_torchcodec_available(), reason="Requires torchcodec")
@pytest.mark.parametrize(
    "path", ["test/fixtures/stereo.opus", "test/fixtures/stereo.webm"]
)
@pytest.mark.parametrize("header_start", [None, 0.032])
def test_torchcodec_read_opus_ignores_header_start(path, header_start, monkeypatch):
    # With FFmpeg 8.1, stereo.webm reports a header start of 0.032 s while its
    # decoded audio starts at 0. header_start forces that value with any FFmpeg.
    if header_start is not None:
        import torchcodec.decoders

        class Decoder(torchcodec.decoders.AudioDecoder):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.metadata.begin_stream_seconds_from_header = header_start

        monkeypatch.setattr(torchcodec.decoders, "AudioDecoder", Decoder)
    backend = TorchcodecBackend()
    full, sr = backend.read_audio(path)
    audio, _ = backend.read_audio(path, offset=0.0, duration=0.2)
    assert audio.shape == (2, round(0.2 * sr))
    np.testing.assert_array_equal(audio, full[:, : audio.shape[1]])


def test_torchcodec_not_applicable_for_sph():
    backend = TorchcodecBackend()
    assert not backend.is_applicable("test/fixtures/stereo.sph")


@pytest.mark.skipif(not is_torchcodec_available(), reason="Requires torchcodec")
@pytest.mark.parametrize("format", ["wav", "flac", "mp3", "m4a"])
def test_torchcodec_save_and_load(tmp_path, format):
    backend = TorchcodecBackend()
    assert backend.supports_save()
    audio = np.clip(np.random.randn(1, 16000).astype(np.float32) * 0.5, -1.0, 1.0)
    path = tmp_path / f"test.{format}"
    backend.save_audio(path, audio, sampling_rate=16000, format=format)
    restored, sr = backend.read_audio(path)
    assert sr == 16000
    assert restored.shape[0] == 1
    if format in ("wav", "flac"):
        np.testing.assert_allclose(audio, restored, atol=1e-4)


@pytest.mark.skipif(not is_torchcodec_available(), reason="Requires torchcodec")
def test_torchcodec_audio_backend_contextmanager():
    with lhotse.audio_backend("TorchcodecBackend") as b:
        assert isinstance(b, TorchcodecBackend)


@pytest.mark.skipif(not is_torchcodec_available(), reason="Requires torchcodec")
def test_video_file_falls_back_when_torchaudio_has_no_ffmpeg(monkeypatch):
    # torchaudio>=2.9 is importable but no longer provides torchaudio.io
    monkeypatch.setattr(lhotse.audio.backend, "is_torchaudio_available", lambda: True)
    monkeypatch.setattr(
        lhotse.audio.backend, "torchaudio_ffmpeg_backend_available", lambda: False
    )
    path = Path("test/fixtures/big_buck_bunny_small.mp4")
    with lhotse.audio_backend("default"):
        recording = Recording.from_file(path)
        audio = recording.load_audio()
    assert audio.shape == (6, recording.num_samples)
    assert recording.video == VideoInfo(fps=25.0, num_frames=132, height=240, width=320)
    assert not TorchaudioFFMPEGBackend().handles_special_case(path)
    assert TorchcodecBackend().handles_special_case(path)
