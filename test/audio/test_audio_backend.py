import pytest

import lhotse
from lhotse.audio.backend import CompositeAudioBackend, LibsndfileBackend


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
