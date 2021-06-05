import warnings
from dataclasses import asdict, dataclass
from decimal import ROUND_HALF_UP
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torchaudio
from packaging.version import parse as _version

from lhotse.utils import Seconds, compute_num_samples, during_docs_build, perturb_num_samples

if not during_docs_build() and _version(torchaudio.__version__) < _version('0.7'):
    warnings.warn('Torchaudio SoX effects chains are only introduced in version 0.7 - '
                  'please upgrade your PyTorch to 1.7+ and torchaudio to 0.7+ to use them.')


@dataclass
class RandomValue:
    """
    Represents a uniform distribution in the range [start, end).
    """
    start: Union[int, float]
    end: Union[int, float]

    def sample(self) -> float:
        return np.random.uniform(self.start, self.end)


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
        >>>    ['reverb', 50, 50, RandomValue(0, 100)],
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
        assert augmented.shape[0] == tensor.shape[0], "Lhotse does not support modifying the number " \
                                                      "of channels during data augmentation."
        assert sampling_rate == new_sampling_rate, \
            f"Lhotse does not support changing the sampling rate during data augmentation. " \
            f"The original SR was '{sampling_rate}', after augmentation it's '{new_sampling_rate}'."
        # Matching shapes after augmentation -> early return.
        if augmented.shape[1] == tensor.shape[1]:
            return augmented
        # We will truncate/zero-pad the signal if the number of samples has changed to mimic
        # the WavAugment behavior that we relied upon so far.
        resized = torch.zeros_like(tensor)
        if augmented.shape[1] > tensor.shape[1]:
            resized = augmented[:, :tensor.shape[1]]
        else:
            resized[:, :augmented.shape[1]] = augmented
        return resized

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


class AudioTransform:
    """
    Base class for all audio transforms that are going to be lazily applied on
    ``Recording`` during loading the audio into memory.

    Any ``AudioTransform`` can be used like a Python function, that expects two arguments:
    a numpy array of samples, and a sampling rate. E.g.:

        >>> fn = AudioTransform.from_dict(...)
        >>> new_audio = fn(audio, sampling_rate)

    Since we often use cuts of the original recording, they will refer to the timestamps
    of the augmented audio (which might be speed perturbed and of different duration).
    Each transform provides a helper method to recover the original audio timestamps:

        >>> # When fn does speed perturbation:
        >>> fn.reverse_timestamps(offset=5.055555, duration=10.1111111, sampling_rate=16000)
        ... (5.0, 10.0)

    Furthermore, ``AudioTransform`` can be easily (de)serialized to/from dict
    that contains its name and parameters.
    This enables storing recording and cut manifests with the transform info
    inside, avoiding the need to store the augmented recording version on disk.

    All audio transforms derived from this class are "automagically" registered,
    so that ``AudioTransform.from_dict()`` can "find" the right type given its name
    to instantiate a specific transform object.
    All child classes are expected to be decorated with a ``@dataclass`` decorator.
    """
    KNOWN_TRANSFORMS = {}

    def __init_subclass__(cls, **kwargs):
        if cls.__name__ not in AudioTransform.KNOWN_TRANSFORMS:
            AudioTransform.KNOWN_TRANSFORMS[cls.__name__] = cls
        super().__init_subclass__(**kwargs)

    def to_dict(self) -> dict:
        data = asdict(self)
        return {'name': type(self).__name__, 'kwargs': data}

    @staticmethod
    def from_dict(data: dict) -> 'AudioTransform':
        assert data['name'] in AudioTransform.KNOWN_TRANSFORMS, f"Unknown transform type: {data['name']}"
        return AudioTransform.KNOWN_TRANSFORMS[data['name']](**data['kwargs'])

    def __call__(self, samples: np.ndarray, sampling_rate: int) -> np.ndarray:
        """
        Apply transform.

        To be implemented in derived classes.
        """
        raise NotImplementedError

    def reverse_timestamps(
            self,
            offset: Seconds,
            duration: Optional[Seconds],
            sampling_rate: int
    ) -> Tuple[Seconds, Optional[Seconds]]:
        """
        Convert ``offset`` and ``duration`` timestamps to be adequate for the audio before the transform.
        Useful for on-the-fly augmentation when a particular chunk of audio needs to be read from disk.

        To be implemented in derived classes.
        """
        raise NotImplementedError


@dataclass
class Speed(AudioTransform):
    """
    Speed perturbation effect, the same one as invoked with `sox speed` in the command line.

    It resamples the signal back to the input sampling rate, so the number of output samples will
    be smaller or greater, depending on the speed factor.
    """
    factor: float

    def __call__(self, samples: np.ndarray, sampling_rate: int) -> np.ndarray:
        effect = [['speed', str(self.factor)], ['rate', str(sampling_rate)]]
        if isinstance(samples, np.ndarray):
            samples = torch.from_numpy(samples)
        augmented, new_sampling_rate = torchaudio.sox_effects.apply_effects_tensor(samples, sampling_rate, effect)
        return augmented.numpy()

    def reverse_timestamps(
            self,
            offset: Seconds,
            duration: Optional[Seconds],
            sampling_rate: int
    ) -> Tuple[Seconds, Optional[Seconds]]:
        """
        This method helps estimate the original offset and duration for a recording
        before speed perturbation was applied.
        We need this estimate to know how much audio to actually load from disk during the
        call to ``load_audio()``.
        """
        start_sample = compute_num_samples(offset, sampling_rate)
        num_samples = compute_num_samples(duration, sampling_rate) if duration is not None else None
        start_sample = perturb_num_samples(start_sample, 1 / self.factor)
        num_samples = perturb_num_samples(num_samples, 1 / self.factor) if num_samples is not None else None
        return start_sample / sampling_rate, num_samples / sampling_rate if num_samples is not None else None


@dataclass
class Resample(AudioTransform):
    """
    Resampling effect, the same one as invoked with `sox rate` in the command line.
    """
    source_sampling_rate: int
    target_sampling_rate: int

    def __call__(self, samples: np.ndarray, *args, **kwargs) -> np.ndarray:
        effect = [['rate', str(self.target_sampling_rate)]]
        if isinstance(samples, np.ndarray):
            samples = torch.from_numpy(samples)
        augmented, _ = torchaudio.sox_effects.apply_effects_tensor(samples, self.source_sampling_rate, effect)
        return augmented.numpy()

    def reverse_timestamps(
            self,
            offset: Seconds,
            duration: Optional[Seconds],
            sampling_rate: int
    ) -> Tuple[Seconds, Optional[Seconds]]:
        """
        This method helps estimate the original offset and duration for a recording
        before resampling was applied.
        We need this estimate to know how much audio to actually load from disk during the
        call to ``load_audio()``.

        In case of resampling, the timestamps might change slightly when using non-trivial
        pairs of sampling rates, e.g. 16kHz -> 22.05kHz, because the number of samples in
        the resampled audio might actually correspond to incrementally larger/smaller duration.
        E.g. 16kHz, 235636 samples correspond to 14.72725s duration; after resampling to 22.05kHz,
        it is 324736 samples which correspond to 14.727256235827664s duration.
        """
        old_num_samples = compute_num_samples(offset, self.source_sampling_rate, rounding=ROUND_HALF_UP)
        old_offset = old_num_samples / self.source_sampling_rate
        if duration is not None:
            old_num_samples = compute_num_samples(duration, self.source_sampling_rate, rounding=ROUND_HALF_UP)
            old_duration = old_num_samples / self.source_sampling_rate
        else:
            old_duration = None
        return old_offset, old_duration


def speed(sampling_rate: int) -> List[List[str]]:
    return [
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
