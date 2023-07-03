import warnings
from dataclasses import asdict, dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch

from lhotse.augmentation.transform import AudioTransform
from lhotse.utils import EPSILON, Seconds, is_module_available


@dataclass
class LoudnessNormalization(AudioTransform):
    """
    Loudness normalization based on pyloudnorm: https://github.com/csteinmetz1/pyloudnorm.
    """

    target: float

    def __call__(
        self, samples: Union[np.ndarray, torch.Tensor], sampling_rate: int
    ) -> np.ndarray:
        if torch.is_tensor(samples):
            samples = samples.cpu().numpy()
        augmented = normalize_loudness(
            samples, target=self.target, sampling_rate=sampling_rate
        )
        return augmented

    def reverse_timestamps(
        self, offset: Seconds, duration: Optional[Seconds], sampling_rate: int
    ) -> Tuple[Seconds, Optional[Seconds]]:
        return offset, duration


def normalize_loudness(
    audio: np.ndarray,
    target: float,
    sampling_rate: int = 16000,
) -> np.ndarray:
    """
    Applies pyloudnorm based loudness normalization to the input audio. The input audio
    can have up to 5 channels, with the following order: [left, right, center, left_surround, right_surround]

    :param audio: the input audio, expected to be 2D with shape (channels, samples).
    :param target: the target loudness in LUFS.
    :param sampling_rate: the sampling rate of the audio.
    :return: the loudness normalized audio.
    """
    if not is_module_available("pyloudnorm"):
        raise ImportError(
            "Please install pyloudnorm first using 'pip install pyloudnorm'"
        )

    import pyloudnorm as pyln

    assert audio.ndim == 2, f"Expected 2D audio shape, got: {audio.shape}"

    duration = audio.shape[1] / sampling_rate

    # measure the loudness first
    meter = pyln.Meter(
        sampling_rate, block_size=min(0.4, duration - EPSILON)
    )  # create BS.1770 meter
    loudness = meter.integrated_loudness(audio.T)

    # loudness normalize audio to target LUFS. We will ignore the warnings related to
    # clipping the audio.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loudness_normalized_audio = pyln.normalize.loudness(audio.T, loudness, target)

    return loudness_normalized_audio.T
