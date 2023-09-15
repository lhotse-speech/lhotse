from pathlib import Path

import pytest
import torch

from lhotse import Recording

COLOR = 3
HEIGHT = 240
WIDTH = 320
FPS = 25.0
FRAMES_TOTAL = 132
AUDIO_CHANNELS = 6


@pytest.fixture
def audio_path() -> Path:
    return Path("test/fixtures/stereo.wav")


def test_audio_recording_has_no_video(audio_path):
    assert not Recording.from_file(audio_path).has_video


def test_video_recording_from_video_file(video_path):
    video_recording = Recording.from_file(video_path)
    assert video_recording.duration == 5.28
    assert video_recording.has_video
    assert video_recording.video.duration == 5.28
    assert video_recording.video.fps == FPS
    assert video_recording.video.num_frames == FRAMES_TOTAL
    assert video_recording.video.height == HEIGHT
    assert video_recording.video.width == WIDTH
    assert video_recording.video.frame_length == 0.04


@pytest.mark.parametrize("with_audio", [True, False])
def test_video_recording_load_video(video_recording, with_audio):
    video, audio = video_recording.load_video(with_audio=with_audio)

    assert video.dtype == torch.uint8

    expected_dims = (132, COLOR, HEIGHT, WIDTH)
    assert video.shape == expected_dims

    if with_audio:
        assert audio.dtype == torch.float32
        assert audio.shape == (AUDIO_CHANNELS, 253440)


def test_video_recording_load_video_rescaled(video_recording):
    video_recording = video_recording.with_video_resolution(width=640, height=480)

    video, audio = video_recording.load_video(with_audio=False)

    expected_dims = (132, COLOR, 480, 640)
    assert video.shape == expected_dims


@pytest.mark.xfail(reason="move_to_memory is not implemented yet for video")
def test_video_recording_move_to_memory(video_recording):
    memory_recording = video_recording.move_to_memory()

    video, audio = video_recording.load_video()
    video1, audio1 = memory_recording.load_video()

    torch.testing.assert_close(video, video1)
    torch.testing.assert_close(audio, audio1)


def test_video_recording_load_video_consistent_audio_duration(video_recording):
    assert video_recording.num_samples == 253440

    # audio would have 254976 in reality, but we truncated it to match the video duration
    audio_full = video_recording.load_audio()
    assert audio_full.shape[1] == 253440

    video, audio = video_recording.load_video()
    # we truncated the audio when loading through load_video
    assert (
        round(
            video.shape[0] / video_recording.video.fps * video_recording.sampling_rate
        )
        == 253440
    )
    # audio and video duration is the same when loading audio through load_video
    assert audio.shape[1] / video_recording.sampling_rate == pytest.approx(
        video.shape[0] / video_recording.video.fps
    )


@pytest.mark.parametrize("num_frames", [1, 2, 131, 132])
def test_video_recording_load_video_offset(video_recording, num_frames):
    offset = 0.04 * num_frames  # fps=25 <=> frame_dur = 0.04

    video, _ = video_recording.load_video(
        offset=offset, duration=None, with_audio=False
    )

    expected_dims = (132 - num_frames, COLOR, HEIGHT, WIDTH)
    assert video.shape == expected_dims


def test_video_recording_load_video_duration(video_recording):
    video, _ = video_recording.load_video(duration=2.0, with_audio=False)

    expected_dims = (25 * 2, COLOR, HEIGHT, WIDTH)
    assert video.shape == expected_dims

    full_video, _ = video_recording.load_video(with_audio=False)

    torch.testing.assert_close(video, full_video[:50])


def test_video_recording_load_video_over_duration(video_recording):
    video, _ = video_recording.load_video(duration=10.0, with_audio=False)

    expected_dims = (132, COLOR, HEIGHT, WIDTH)
    assert video.shape == expected_dims

    full_video, _ = video_recording.load_video(with_audio=False)

    torch.testing.assert_close(video, full_video)
