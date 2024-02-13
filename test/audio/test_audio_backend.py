from io import BytesIO

import numpy as np
import pytest

import lhotse
from lhotse.audio import AudioLoadingError
from lhotse.audio.backend import (
    CompositeAudioBackend,
    LibsndfileBackend,
    TorchaudioDefaultBackend,
    TorchaudioFFMPEGBackend,
    check_torchaudio_version_gt,
    torchaudio_ffmpeg_backend_available,
    torchaudio_soundfile_supports_format,
)
from lhotse.testing.random import deterministic_rng
from lhotse.utils import INT16MAX


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
