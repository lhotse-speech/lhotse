import warnings
from typing import List, Union

import numpy as np
import torch

__all__ = ['is_wav_augment_available', 'WavAugmenter', 'available_wav_augmentations', 'register_wav_augmentation',
           'pitch', 'reverb', 'pitch_reverb_tdrop']


def is_wav_augment_available() -> bool:
    """Returns a boolean indicating if WavAugment is both installed and possible to import."""
    try:
        import augment
        return True
    except:
        return False


class WavAugmenter:
    """
    A wrapper class for WavAugment's effect chain.
    You should construct the ``augment.EffectChain`` beforehand and pass it on to this class.

    This class is only available when WavAugment is installed, as it is an optional dependency for Lhotse.
    It can be installed using the script in "<main-repo-directory>/tools/install_wavaugment.sh"

    For more details on how to augment, see https://github.com/facebookresearch/WavAugment
    """

    def __init__(self, effect_chain):
        warnings.warn('WavAugment support is deprecated and it will eventually be removed from Lhotse. '
                      'For similar functionality, please use torchaudio based augmentation in '
                      '"lhotse.augmentation.torchaudio". It requires PyTorch 1.7+ and torchaudio 0.7+.',
                      category=DeprecationWarning)
        # A local import so that ``augment`` can be optional.
        import augment
        self.chain: augment.EffectChain = effect_chain

    @staticmethod
    def create_predefined(name: str, sampling_rate: int, **kwargs) -> 'WavAugmenter':
        """
        Create a WavAugmenter class with one of the predefined augmentation setups available in Lhotse.
        Some examples are: "pitch", "reverb", "pitch_reverb_tdrop".

        :param name: the name of the augmentation setup.
        :param sampling_rate: expected sampling rate of the input audio.
        """
        return WavAugmenter(
            effect_chain=_DEFAULT_AUGMENTATIONS[name](sampling_rate=sampling_rate, **kwargs),
        )

    def __call__(
            self,
            audio: Union[torch.Tensor, np.ndarray],
            sampling_rate: int
    ) -> np.ndarray:
        """
        Apply the effect chain on the ``audio`` tensor.

        :param audio: a (num_channels, num_samples) shaped tensor placed on the CPU.
        :param sampling_rate: The input and output sampling rate (has to be the same).
        :return a numpy ndarray with the augmented audio signal.
            In case SoX returned Nan or Inf for some sample, fall back to returning the non-augmented
            signal instead.
        """
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)

        assert len(audio.shape) == 2, f"Expected 2-dim tensor with shape (num_channels, num_samples), got {audio.shape}"

        augmented = self.chain.apply(
            input_tensor=audio,
            src_info={
                'channels': audio.shape[0],
                'length': audio.shape[1],
                'rate': sampling_rate,
            },
            target_info={
                'channels': 1,
                'length': audio.shape[1],
                'rate': sampling_rate,
            }
        )

        # Sox might misbehave sometimes by giving nan/inf if sequences are too short (or silent)
        # and the effect chain includes eg `pitch`
        if torch.isnan(augmented).any() or torch.isinf(augmented).any():
            warnings.warn('NaN/Inf encountered in augmented sox output - returning non-augmented audio.')
            return audio.numpy()
        return augmented.numpy()


_DEFAULT_AUGMENTATIONS = {}


def available_wav_augmentations() -> List[str]:
    """Return a list of augmentation setups available in Lhotse."""
    return list(_DEFAULT_AUGMENTATIONS.keys())


def register_wav_augmentation(fn):
    """
    Register a new augmentation setup that can be retrieved by the name of the function passed in.

    :param fn: a function returning an ``augment.EffectChain`` that performs wav augmentation.
    """
    _DEFAULT_AUGMENTATIONS[fn.__name__] = fn
    return fn


@register_wav_augmentation
def pitch(sampling_rate: int):
    """
    Returns a pitch modification effect for wav augmentation.

    :param sampling_rate: a sampling rate value for which the effect will be created (resampling is needed for pitch).
    """
    import augment
    effect_chain = augment.EffectChain()
    # The pitch effect changes the sampling ratio; we have to compensate for that.
    # Here, we specify 'quick' options on both pitch and rate effects, to speed up things
    effect_chain.pitch("-q", _random_pitch_shift).rate("-q", sampling_rate)
    return effect_chain


@register_wav_augmentation
def speed(sampling_rate: int):
    """
    Returns a pitch modification effect for wav augmentation.

    :param sampling_rate: a sampling rate value for which the effect will be created (resampling is needed for pitch).
    """
    import augment
    effect_chain = augment.EffectChain()
    # The pitch effect changes the sampling ratio; we have to compensate for that.
    # Here, we specify 'quick' options on both pitch and rate effects, to speed up things
    effect_chain.speed(_random_speed_perturb).rate("-q", sampling_rate)
    return effect_chain


@register_wav_augmentation
def reverb(*args, **kwargs):
    """
    Returns a reverb effect for wav augmentation.
    """
    import augment
    effect_chain = augment.EffectChain()
    # Reverb it makes the signal to have two channels,
    # which we combine into 1 by running `channels` w/o parameters
    effect_chain.reverb(50, 50, _random_room_size).channels()
    return effect_chain


@register_wav_augmentation
def pitch_reverb_tdrop(sampling_rate: int):
    """
    Returns an effect chain composed of pitch modification, reverberation and time dropout proposed in:

    * https://github.com/facebookresearch/WavAugment/blob/master/examples/python/librispeech_selfsupervised.py#L152
    * https://arxiv.org/abs/2007.00991

    :param sampling_rate: a sampling rate value for which the effect will be created (resampling is needed for pitch).
    """
    import augment
    effect_chain = augment.EffectChain()
    # The pitch effect changes the sampling ratio; we have to compensate for that.
    # Here, we specify 'quick' options on both pitch and rate effects, to speed up things
    effect_chain.pitch("-q", _random_pitch_shift).rate("-q", sampling_rate)
    # Next effect we add is `reverb`; it adds makes the signal to have two channels,
    # which we combine into 1 by running `channels` w/o parameters
    effect_chain.reverb(50, 50, _random_room_size).channels()
    # Futher, we add an effect that randomly drops one 50ms subsequence
    effect_chain.time_dropout(max_seconds=50 / 1000)
    return effect_chain


def _random_speed_perturb() -> int:
    """The returned values are speed perturbation factors (0.9x - 1.1x the original speed)."""
    return np.random.uniform(0.9, 1.1)


def _random_pitch_shift() -> int:
    """The returned values are 1/100ths of a semitone, meaning the default is up to a minor third shift up or down."""
    return np.random.randint(-300, 300)


def _random_room_size() -> int:
    """The returned values correspond to the 'room scale' parameter of SoX."""
    return np.random.randint(0, 100)
