import pytest
import torch

from lhotse import CutSet
from lhotse.dataset.signal_transforms import GlobalMVN
from lhotse.dataset.speech_synthesis import SpeechSynthesisDataset


@pytest.fixture
def cut_set():
    return CutSet.from_json("test/fixtures/ljspeech/cuts.json")


@pytest.mark.parametrize("transform", [None, GlobalMVN, [GlobalMVN]])
def test_speech_synthesis_dataset(cut_set, transform):
    if isinstance(transform, list):
        transform = [transform[0].from_cuts(cut_set)]
    elif isinstance(transform, GlobalMVN):
        transform = transform(cut_set)
    else:
        transform = None

    dataset = SpeechSynthesisDataset(feature_transforms=transform)
    example = dataset[cut_set]
    assert example["audio"].shape[1] > 0
    assert example["features"].shape[1] > 0
    assert len(example["text"]) > 0
    assert len(example["text"][0]) > 0

    assert example["audio"].ndim == 2
    assert example["features"].ndim == 3

    assert isinstance(example["audio_lens"], torch.IntTensor)
    assert isinstance(example["features_lens"], torch.IntTensor)

    assert example["audio_lens"].ndim == 1
    assert example["features_lens"].ndim == 1
