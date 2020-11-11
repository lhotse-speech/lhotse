import math

import pytest
import torch

torchaudio = pytest.importorskip('torchaudio', minversion='0.6')

from lhotse.augmentation import SoxEffectTransform, pitch, reverb, speed


@pytest.fixture
def audio():
    return torch.sin(2 * math.pi * torch.linspace(0, 1, 16000)).unsqueeze(0).numpy()


@pytest.mark.parametrize('effect', [reverb, pitch, speed])
def test_example_augmentation(audio, effect):
    augment_fn = SoxEffectTransform(effects=effect(16000))
    augmented_audio = augment_fn(audio, sampling_rate=16000)
    assert augmented_audio.shape == audio.shape
    assert augmented_audio != audio
