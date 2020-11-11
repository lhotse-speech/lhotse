import random
from dataclasses import dataclass
from typing import List, Union

import numpy as np
import torch
import torchaudio


@dataclass
class RandomValue:
    """
    Represents a uniform distribution in the range [start, end].
    """
    start: Union[int, float]
    end: Union[int, float]

    def sample(self):
        return random.uniform(self.start, self.end)


# Input to the SoxEffectTransform class - the values are either effect names,
# numeric parameters, or uniform distribution over possible values.
EffectsList = List[List[Union[str, int, float, RandomValue]]]


class SoxEffectTransform:
    """
    Class-style wrapper for torchaudio SoX effect chains.
    It should be initialized with a config-like list of items that define SoX effect to be applied.
    It supports sampling randomized values for effect parameters through the ``RandomValue`` wrapper.

    Example:
        >>> audio = np.random.rand(16000)
        >>> augment_fn = SoxEffectTransform(effects=[
        >>>    ['reverb', 50, 50, RandomValue(0, 100)],  #
        >>>    ['speed', RandomValue(0.9, 1.1)],
        >>>    ['rate', 16000],
        >>> ])
        >>> augmented = augment_fn(audio, 16000)

    See SoX manual or ``torchaudio.sox_effects.effect_names()`` for the list of possible effects.
    The parameters and the meaning of the values are explained in SoX manual/help.
    """

    def __init__(self, effects: EffectsList):
        super().__init__()
        self.effects = effects

    def __call__(self, tensor: Union[torch.Tensor, np.ndarray], sampling_rate: int):
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        effects = self.sample_effects()
        augmented, new_sampling_rate = torchaudio.sox_effects.apply_effects_tensor(tensor, sampling_rate, effects)
        assert sampling_rate == new_sampling_rate, \
            f"Lhotse does not support changing the sampling rate during data augmentation. " \
            f"The original SR was '{sampling_rate}', after augmentation it's '{new_sampling_rate}'."
        return augmented

    def sample_effects(self) -> List[List[str]]:
        """
        Resolve a list of effects, replacing random distributions with samples from them.
        It converts every number to string to match the expectations of torchaudio.
        """
        return [
            [
                str(item.sample() if isinstance(item, RandomValue) else item)
                for item in effect
            ]
            for effect in self.effects
        ]


def speed(sampling_rate: int) -> List[List[str]]:
    return [
        # Random speed perturbation factor between 0.9x and 1.1x the original speed
        ['speed', RandomValue(0.9, 1.1)],
        ['rate', sampling_rate],  # Resample back to the original sampling rate (speed changes it)
    ]


def reverb(sampling_rate: int) -> List[List[str]]:
    return [
        ['reverb', 50, 50, RandomValue(0, 100)],
        ['remix', '-'],  # Merge all channels (reverb changes mono to stereo)
    ]


def pitch(sampling_rate: int) -> List[List[str]]:
    return [
        # The returned values are 1/100ths of a semitone, meaning the default is up to a minor third shift up or down.
        ['pitch', '-q', RandomValue(-300, 300)],
        ['rate', sampling_rate]  # Resample back to the original sampling rate (pitch changes it)
    ]
