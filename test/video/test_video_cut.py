import pytest
import torch

from lhotse import MultiCut

COLOR = 3
HEIGHT = 240
WIDTH = 320
FPS = 25.0
FRAMES_TOTAL = 132
AUDIO_CHANNELS = 6


@pytest.fixture(scope="session")
def video_cut(video_recording) -> MultiCut:
    return video_recording.to_cut()


def test_video_multi_cut(video_cut):
    assert video_cut.has_video
    assert video_cut.has("video")
    assert video_cut.num_channels == AUDIO_CHANNELS
    assert video_cut.video.fps == FPS
    assert video_cut.video.width == WIDTH
    assert video_cut.video.height == HEIGHT
    assert video_cut.video.num_frames == FRAMES_TOTAL

    # Load all audio channels with video
    video, audio = video_cut.load_video()
    assert video.dtype == torch.uint8
    assert video.shape == (132, COLOR, HEIGHT, WIDTH)
    assert audio.dtype == torch.float32
    assert audio.shape == (AUDIO_CHANNELS, 253440)

    # Load one audio channel with video
    video, audio = video_cut.load_video(channel=0)
    assert video.dtype == torch.uint8
    assert video.shape == (132, COLOR, HEIGHT, WIDTH)
    assert audio.dtype == torch.float32
    assert audio.shape == (1, 253440)


def test_video_mono_cut(video_cut):
    video_cut = video_cut.to_mono()[0]
    assert video_cut.has_video
    assert video_cut.has("video")
    assert video_cut.num_channels == 1
    assert video_cut.video.fps == FPS
    assert video_cut.video.width == WIDTH
    assert video_cut.video.height == HEIGHT
    assert video_cut.video.num_frames == FRAMES_TOTAL

    video, audio = video_cut.load_video()
    assert video.dtype == torch.uint8
    assert video.shape == (132, COLOR, HEIGHT, WIDTH)
    assert audio.dtype == torch.float32
    assert audio.shape == (1, 253440)


def test_video_mixed_cut_from_padding(video_cut):

    orig_video, orig_audio = video_cut.load_video()

    video_cut = video_cut.pad(duration=10.0)
    assert video_cut.has_video
    assert video_cut.video.fps == FPS
    assert video_cut.video.width == WIDTH
    assert video_cut.video.height == HEIGHT
    assert video_cut.video.num_frames == 25 * 10

    # Load all audio channels with video
    video, audio = video_cut.load_video()
    assert video.dtype == torch.uint8
    assert video.shape == (25 * 10, COLOR, HEIGHT, WIDTH)
    assert audio.dtype == torch.float32
    assert audio.shape == (AUDIO_CHANNELS, 48000 * 10)

    torch.testing.assert_close(video[:132], orig_video)
    assert (video[132:] == 0).all()
    torch.testing.assert_close(audio[:, :253440], orig_audio)
    assert (audio[:, 253440:] == 0).all()


def test_video_mixed_cut_from_appending(video_cut):

    orig_video, orig_audio = video_cut.load_video()

    video_cut = video_cut.append(video_cut)
    assert video_cut.has_video
    assert video_cut.video.fps == FPS
    assert video_cut.video.width == WIDTH
    assert video_cut.video.height == HEIGHT
    assert video_cut.video.num_frames == 132 * 2

    # Load all audio channels with video
    video, audio = video_cut.load_video()
    assert video.dtype == torch.uint8
    assert video.shape == (132 * 2, COLOR, HEIGHT, WIDTH)
    assert audio.dtype == torch.float32
    assert audio.shape == (AUDIO_CHANNELS, 253440 * 2)

    torch.testing.assert_close(video[:132], orig_video)
    torch.testing.assert_close(video[132:], orig_video)
    torch.testing.assert_close(audio[:, :253440], orig_audio)
    torch.testing.assert_close(audio[:, 253440:], orig_audio)
