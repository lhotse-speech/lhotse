import warnings
from typing import Union

import numpy as np
import torch


def default_effect_chain(sampling_rate):
    """
    Returns an effect chain composed of pitch modification, reverberation and time dropout proposed in:
    https://github.com/facebookresearch/WavAugment/blob/master/examples/python/librispeech_selfsupervised.py#L152
    https://arxiv.org/abs/2007.00991
    """

    def random_pitch_shift():
        return np.random.randint(-300, 300)

    def random_room_size():
        return np.random.randint(0, 100)

    import augment
    effect_chain = augment.EffectChain()
    # The pitch effect changes the sampling ratio; we have to compensate for that.
    # Here, we specify 'quick' options on both pitch and rate effects, to speed up things
    effect_chain.pitch("-q", random_pitch_shift).rate("-q", sampling_rate)
    # Next effect we add is `reverb`; it adds makes the signal to have two channels,
    # which we combine into 1 by running `channels` w/o parameters
    effect_chain.reverb(50, 50, random_room_size).channels()
    # Futher, we add an effect that randomly drops one 50ms subsequence
    effect_chain.time_dropout(max_seconds=50 / 1000)
    return effect_chain


class WavAugmenter:
    """
    A wrapper class for WavAugment's effect chain.
    You should construct the ``augment.EffectChain`` beforehand and pass it on to this class.
    For more details on how to augment, see https://github.com/facebookresearch/WavAugment
    """

    def __init__(self, effect_chain):
        # A local import so that ``augment`` can be optional.
        import augment
        self.chain: augment.EffectChain = effect_chain

    def apply(self, audio: Union[torch.Tensor, np.ndarray], sampling_rate: int) -> torch.Tensor:
        """Apply the effect chain on the ``audio`` tensor. Must be placed on the CPU."""
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)

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
            return audio.clone()
        return augmented
