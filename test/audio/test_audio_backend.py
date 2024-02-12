import numpy as np
import pytest

import lhotse
from lhotse.audio.backend import (
    CompositeAudioBackend,
    LibsndfileBackend,
    TorchaudioDefaultBackend,
    TorchaudioFFMPEGBackend,
    torchaudio_2_0_ffmpeg_enabled,
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


@pytest.mark.parametrize("backend", lhotse.available_audio_backends())
@pytest.mark.parametrize("format", ["wav", "flac", "opus"])
def test_save_and_load(deterministic_rng, tmp_path, backend, format):
    path = tmp_path / f"test.{format}"

    if backend == "CompositeAudioBackend":
        return
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
            if backend == "LibsndfileBackend":
                assert restored.shape == audio.shape
            else:
                assert restored.shape == (
                    1,
                    48000,
                )  # only libnsdfile auto-resamples OPUS


@pytest.mark.parametrize(
    ["backend_save", "backend_read"],
    [
        pytest.param(
            LibsndfileBackend,
            TorchaudioFFMPEGBackend,
            marks=pytest.mark.skipif(
                not torchaudio_2_0_ffmpeg_enabled(),
                reason="Requires Torchaudio + FFMPEG",
            ),
        ),
        (LibsndfileBackend, TorchaudioDefaultBackend),
        (TorchaudioDefaultBackend, LibsndfileBackend),
        pytest.param(
            TorchaudioDefaultBackend,
            TorchaudioFFMPEGBackend,
            marks=pytest.mark.skipif(
                not torchaudio_2_0_ffmpeg_enabled(),
                reason="Requires Torchaudio + FFMPEG",
            ),
        ),
        (TorchaudioFFMPEGBackend, LibsndfileBackend),
        pytest.param(
            TorchaudioFFMPEGBackend,
            TorchaudioDefaultBackend,
            marks=pytest.mark.skipif(
                not torchaudio_2_0_ffmpeg_enabled(),
                reason="Requires Torchaudio + FFMPEG",
            ),
        ),
    ],
)
def test_save_load_opus_different_backends(
    deterministic_rng, tmp_path, backend_save, backend_read
):
    with lhotse.audio_backend(LibsndfileBackend):
        audio = (np.random.randint(0, INT16MAX, size=(1, 16000)) / INT16MAX).astype(
            np.float32
        )
        path = tmp_path / "test.opus"
        lhotse.audio.backend.save_audio(path, audio, sampling_rate=16000, format="opus")

        recording_old = lhotse.Recording.from_file(path)

    with lhotse.audio_backend(TorchaudioFFMPEGBackend):
        # Raw read/save utilities work across backends
        restored, sr = lhotse.audio.backend.read_audio(path)
        assert restored.shape == (1, 48000)

        # lhotse Recording works when created with a different backend
        restored2 = recording_old.load_audio()
        np.testing.assert_allclose(restored2, restored)
        # transcoding does not raise exception
        recording_old.to_cut().truncate(duration=0.2).move_to_memory(audio_format="wav")

        # lhotse Recording works when created with the same backend but data saved using different backend
        recording_new = lhotse.Recording.from_file(path)
        restored3 = recording_new.load_audio()
        np.testing.assert_allclose(restored3, restored)
        # transcoding does not raise exception
        recording_new.to_cut().truncate(duration=0.2).move_to_memory(audio_format="wav")
