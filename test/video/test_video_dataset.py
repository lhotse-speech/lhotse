import pytest
from torch.utils.data import DataLoader

from lhotse import CutSet, MultiCut
from lhotse.dataset import DynamicCutSampler
from lhotse.dataset.collation import collate_video
from lhotse.dataset.video import UnsupervisedAudioVideoDataset

COLOR = 3
HEIGHT = 240
WIDTH = 320
FPS = 25.0
FRAMES = 132
AUDIO_CHANNELS = 6


@pytest.fixture(scope="session")
def video_cut(video_recording) -> MultiCut:
    return video_recording.to_cut()


@pytest.fixture(scope="session")
def video_cut_set(video_cut) -> CutSet:
    return (
        CutSet.from_cuts([video_cut])
        .resample(16000)
        .cut_into_windows(duration=1.0, hop=0.48)
        .filter(lambda c: c.duration > 1 / FPS)
        .repeat(100)
    )


def test_collate_video(video_cut):
    cuts = CutSet.from_cuts([video_cut]).repeat(2)
    video, video_lens, audio, audio_lens = collate_video(cuts)
    assert video.shape == (2, FRAMES, COLOR, HEIGHT, WIDTH)
    assert video_lens.tolist() == [FRAMES, FRAMES]
    assert audio.shape == (2, AUDIO_CHANNELS, 253440)
    assert audio_lens.tolist() == [253440, 253440]


def test_video_dataloading(video_cut_set):
    dataset = UnsupervisedAudioVideoDataset()
    sampler = DynamicCutSampler(video_cut_set, max_duration=2.0, shuffle=True)
    dloader = DataLoader(dataset, sampler=sampler, batch_size=None)

    for step, batch in enumerate(dloader):
        if step == 10:
            break

        for k in "cuts video audio video_lens audio_lens".split():
            assert k in batch

        # Mostly just test that it runs without exceptions for a few steps.
