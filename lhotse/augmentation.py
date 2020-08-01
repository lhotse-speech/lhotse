import warnings
from typing import Union

import numpy as np
import torch


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
