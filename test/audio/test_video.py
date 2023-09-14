from pathlib import Path

import pytest
import torch

from lhotse import Recording

COLOR = 3
HEIGHT = 240
WIDTH = 320


@pytest.fixture
def video_path() -> Path:
    return Path("test/fixtures/big_buck_bunny_small.mp4")


@pytest.fixture
def audio_path() -> Path:
    return Path("test/fixtures/stereo.wav")


def test_recording_has_no_video(audio_path):
    assert not Recording.from_file(audio_path).has_video


def test_recording_from_video_file(video_path):
    # Note: unfortunately audio and video *can* have a different duration.
    recording = Recording.from_file(video_path)
    assert recording.duration == 5.312
    assert recording.has_video
    assert recording.video.duration == 5.28
    assert recording.video.fps == 25.0
    assert recording.video.num_frames == 132
    assert recording.video.height == HEIGHT
    assert recording.video.width == WIDTH


def test_recording_load_video(video_path):
    recording = Recording.from_file(video_path)
    video = recording.load_video()

    assert video.dtype == torch.uint8

    expected_dims = (132, COLOR, HEIGHT, WIDTH)
    assert video.shape == expected_dims


@pytest.mark.parametrize("num_frames", [1, 2, 131, 132])
def test_recording_load_video_offset(video_path, num_frames):
    offset = 0.04 * num_frames  # fps=25 <=> frame_dur = 0.04

    recording = Recording.from_file(video_path)
    video = recording.load_video(offset=offset, duration=None)

    assert video.dtype == torch.uint8

    expected_dims = (132 - num_frames, COLOR, HEIGHT, WIDTH)
    assert video.shape == expected_dims


def test_recording_load_video_duration(video_path):
    recording = Recording.from_file(video_path)
    video = recording.load_video(duration=2.0)

    assert video.dtype == torch.uint8

    expected_dims = (25 * 2, COLOR, HEIGHT, WIDTH)
    assert video.shape == expected_dims

    full_video = recording.load_video()

    torch.testing.assert_close(video, full_video[:50])


def test_recording_load_video_over_duration(video_path):
    recording = Recording.from_file(video_path)
    video = recording.load_video(duration=10.0)

    assert video.dtype == torch.uint8

    expected_dims = (132, COLOR, HEIGHT, WIDTH)
    assert video.shape == expected_dims

    full_video = recording.load_video()

    torch.testing.assert_close(video, full_video)
