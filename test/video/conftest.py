from pathlib import Path

import pytest

from lhotse import Recording
from lhotse.audio.backend import (
    torchaudio_ffmpeg_backend_available,
    torchcodec_supports_video,
)

# Video tests require either torchcodec or torchaudio with the FFMPEG backend.
collect_ignore = []
if not (torchcodec_supports_video() or torchaudio_ffmpeg_backend_available()):
    collect_ignore_glob = ["test_video_*.py"]


@pytest.fixture(scope="session")
def video_path() -> Path:
    return Path("test/fixtures/big_buck_bunny_small.mp4")


@pytest.fixture(scope="session")
def video_recording(video_path) -> Recording:
    return Recording.from_file(video_path)
