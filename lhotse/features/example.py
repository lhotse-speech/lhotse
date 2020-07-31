from dataclasses import dataclass

import numpy as np
from scipy.signal import stft

from lhotse.features import FeatureExtractor
from lhotse.utils import Seconds


@dataclass
class ExampleFeatureExtractorConfig:
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
        f, t, Zxx = stft(samples, sampling_rate, noverlap=round(self.frame_shift * sampling_rate))
        return np.abs(Zxx)

    @property
    def frame_shift(self) -> Seconds:
        return self.config.frame_shift
