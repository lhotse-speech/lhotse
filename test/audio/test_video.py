from pathlib import Path

import pytest
import torch

from lhotse import Recording

COLOR = 3
HEIGHT = 240
WIDTH = 320
AUDIO_CHANNELS = 6


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
    assert recording.video.frame_length == 0.04


@pytest.mark.parametrize("with_audio", [True, False])
def test_recording_load_video(video_path, with_audio):
    recording = Recording.from_file(video_path)
    video, audio = recording.load_video(with_audio=with_audio)

    assert video.dtype == torch.uint8

    expected_dims = (132, COLOR, HEIGHT, WIDTH)
    assert video.shape == expected_dims

    if with_audio:
        assert audio.dtype == torch.float32
        assert audio.shape == (AUDIO_CHANNELS, 253440)


def test_recording_load_video_consistent_audio_duration(video_path):
    recording = Recording.from_file(video_path)

    assert recording.num_samples == 254976

    # audio has full duration when loaded through load_audio
    audio_full = recording.load_audio()
    assert audio_full.shape[1] == 254976

    video, audio = recording.load_video()
    # we truncated the audio when loading through load_video
    assert (
        round(video.shape[0] / recording.video.fps * recording.sampling_rate) == 253440
    )
    # audio and video duration is the same when loading audio through load_video
    assert audio.shape[1] / recording.sampling_rate == pytest.approx(
        video.shape[0] / recording.video.fps
    )


@pytest.mark.parametrize("num_frames", [1, 2, 131, 132])
def test_recording_load_video_offset(video_path, num_frames):
    offset = 0.04 * num_frames  # fps=25 <=> frame_dur = 0.04

    recording = Recording.from_file(video_path)
    video, _ = recording.load_video(offset=offset, duration=None, with_audio=False)

    expected_dims = (132 - num_frames, COLOR, HEIGHT, WIDTH)
    assert video.shape == expected_dims


def test_recording_load_video_duration(video_path):
    recording = Recording.from_file(video_path)
    video, _ = recording.load_video(duration=2.0, with_audio=False)

    expected_dims = (25 * 2, COLOR, HEIGHT, WIDTH)
    assert video.shape == expected_dims

    full_video, _ = recording.load_video(with_audio=False)

    torch.testing.assert_close(video, full_video[:50])


def test_recording_load_video_over_duration(video_path):
    recording = Recording.from_file(video_path)
    video, _ = recording.load_video(duration=10.0, with_audio=False)

    expected_dims = (132, COLOR, HEIGHT, WIDTH)
    assert video.shape == expected_dims

    full_video, _ = recording.load_video(with_audio=False)

    torch.testing.assert_close(video, full_video)
