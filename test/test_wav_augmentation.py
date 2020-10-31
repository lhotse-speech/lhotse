import math

import pytest
import torch

augment = pytest.importorskip('augment')

from lhotse.augmentation import WavAugmenter


@pytest.fixture
def audio():
    return torch.sin(2 * math.pi * torch.linspace(0, 1, 16000)).unsqueeze(0).numpy()


@pytest.mark.parametrize('name', ['reverb', 'pitch', 'pitch_reverb_tdrop'])
def test_predefined_augmentation_setups(audio, name):
    augmenter = WavAugmenter.create_predefined(name=name, sampling_rate=16000)
    augmented_audio = augmenter.apply(audio)
    assert augmented_audio.shape == audio.shape
    assert (augmented_audio != audio).any()
