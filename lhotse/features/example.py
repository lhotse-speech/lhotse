from dataclasses import dataclass

import numpy as np
from scipy.signal import stft

from lhotse.features import FeatureExtractor
from lhotse.utils import Seconds


@dataclass
class ExampleFeatureExtractorConfig:
    frame_len: Seconds = 0.025
    frame_shift: Seconds = 0.01


# @register_extractor
class ExampleFeatureExtractor(FeatureExtractor):
    """
    A minimal class example, showing how to implement a custom feature extractor in Lhotse.
    Note that the "@register_extractor" decorator is commented out so that this example
    is not visible beyond this file.
    """
    name = 'example-feature-extractor'
    config_type = ExampleFeatureExtractorConfig

    def extract(self, samples: np.ndarray, sampling_rate: int) -> np.ndarray:
        f, t, Zxx = stft(
            samples,
            sampling_rate,
            nperseg=round(self.config.frame_len * sampling_rate),
            noverlap=round(self.frame_shift * sampling_rate)
        )
        # Note: returning a magnitude of the STFT might interact badly with lilcom compression,
        # as it performs quantization of the float values and works best with log-scale quantities.
        # It's advised to turn lilcom compression off, or use log-scale, in such cases.
        return np.abs(Zxx)

    @property
    def frame_shift(self) -> Seconds:
        return self.config.frame_shift

    def feature_dim(self, sampling_rate: int) -> int:
        return (sampling_rate * self.config.frame_len) / 2 + 1
