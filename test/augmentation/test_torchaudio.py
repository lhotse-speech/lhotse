import math

import pytest
import torch

torchaudio = pytest.importorskip('torchaudio', minversion='0.6')

from lhotse.augmentation import SoxEffectTransform, pitch, reverb, speed, Speed
from lhotse import AudioTransform

SAMPLING_RATE = 16000


@pytest.fixture
def audio():
    return torch.sin(2 * math.pi * torch.linspace(0, 1, 16000)).unsqueeze(0).numpy()


@pytest.mark.parametrize('effect', [reverb, pitch, speed])
def test_example_augmentation(audio, effect):
    augment_fn = SoxEffectTransform(effects=effect(SAMPLING_RATE))
    augmented_audio = augment_fn(audio, sampling_rate=SAMPLING_RATE)
    assert augmented_audio.shape == audio.shape
    assert augmented_audio != audio


def test_speed_does_not_change_num_samples(audio):
    augment_fn = SoxEffectTransform(effects=speed(SAMPLING_RATE))
    # Since speed() is not deterministic and between 0.9x - 1.1x, multiple invocations
    # will yield either slower (more samples) or faster (less samples) signal.
    # The truncation/padding is performed inside of SoxEffectTransform so the user should not
    # see these changes.
    for _ in range(10):
        augmented_audio = augment_fn(audio, sampling_rate=SAMPLING_RATE)
        assert augmented_audio.shape == audio.shape
        assert augmented_audio != audio


def test_speed(audio):
    speed = Speed(factor=1.1)
    perturbed = speed(audio, SAMPLING_RATE)
    assert perturbed.shape == (1, 14545)


def test_deserialize_transform(audio):
    speed = AudioTransform.from_dict({'name': 'Speed', 'kwargs': {'factor': 1.1}})
    perturbed = speed(audio, SAMPLING_RATE)
    assert perturbed.shape == (1, 14545)


def test_serialize_deserialize_transform(audio):
    speed_orig = Speed(factor=1.1)
    data = speed_orig.to_dict()
    speed = AudioTransform.from_dict(data)
    perturbed = speed(audio, SAMPLING_RATE)
    assert perturbed.shape == (1, 14545)
