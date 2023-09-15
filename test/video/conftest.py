from pathlib import Path

import pytest

from lhotse import Recording


@pytest.fixture(scope="session")
def video_path() -> Path:
    return Path("test/fixtures/big_buck_bunny_small.mp4")


@pytest.fixture(scope="session")
def video_recording(video_path) -> Recording:
    return Recording.from_file(video_path)
