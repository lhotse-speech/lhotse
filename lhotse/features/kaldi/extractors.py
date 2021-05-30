from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import torch

from lhotse.features.base import FeatureExtractor, register_extractor
from lhotse.features.kaldi.layers import Wav2LogFilterBank, Wav2MFCC
from lhotse.utils import EPSILON, Seconds


@dataclass
class KaldiFbankConfig:
    sampling_rate: int = 16000
    frame_length: Seconds = 0.025
    frame_shift: Seconds = 0.01
    fft_length: int = 512
    remove_dc_offset: bool = True
    preemph_coeff: float = 0.97
    window_type: str = 'povey'
    dither: float = 0.0
    snip_edges: bool = False
    energy_floor: float = EPSILON
    raw_energy: bool = True
    use_energy: bool = False
    use_fft_mag: bool = False
    low_freq: float = 20.0
    high_freq: float = -400.0
    num_filters: int = 80
    norm_filters: bool = False


@register_extractor
class KaldiFbank(FeatureExtractor):
    name = "kaldi-fbank"
    config_type = KaldiFbankConfig

    def __init__(self, config: Optional[KaldiFbankConfig] = None):
        super().__init__(config=config)
        self.extractor = Wav2LogFilterBank(**asdict(self.config))

    @property
    def frame_shift(self) -> Seconds:
        return self.config.frame_shift

    def feature_dim(self, sampling_rate: int) -> int:
        return self.config.num_filters

    def extract(self, samples: np.ndarray, sampling_rate: int) -> np.ndarray:
        assert sampling_rate == self.config.sampling_rate, f"KaldiFbank was instantiated for sampling_rate " \
                                                           f"{self.config.sampling_rate}, but " \
                                                           f"sampling_rate={sampling_rate} was passed to extract()."
        return self.extractor(torch.from_numpy(samples))[0].numpy()

    @staticmethod
    def mix(
            features_a: np.ndarray, features_b: np.ndarray, energy_scaling_factor_b: float
    ) -> np.ndarray:
        return np.log(
            np.maximum(
                # protection against log(0); max with EPSILON is adequate since these are energies (always >= 0)
                EPSILON,
                np.exp(features_a) + energy_scaling_factor_b * np.exp(features_b),
            )
        )

    @staticmethod
    def compute_energy(features: np.ndarray) -> float:
        return float(np.sum(np.exp(features)))


@dataclass
class KaldiMfccConfig:
    sampling_rate: int = 16000
    frame_length: Seconds = 0.025
    frame_shift: Seconds = 0.01
    fft_length: int = 512
    remove_dc_offset: bool = True
    preemph_coeff: float = 0.97
    window_type: str = 'povey'
    dither: float = 0.0
    snip_edges: bool = False
    energy_floor: float = EPSILON
    raw_energy: bool = True
    use_energy: bool = False
    use_fft_mag: bool = False
    low_freq: float = 20.0
    high_freq: float = -400.0
    num_filters: int = 23
    norm_filters: bool = False
    num_ceps: int = 13
    cepstral_lifter: int = 22


@register_extractor
class KaldiMfcc(FeatureExtractor):
    name = "kaldi-mfcc"
    config_type = KaldiMfccConfig

    def __init__(self, config: Optional[KaldiMfccConfig] = None):
        super().__init__(config=config)
        self.extractor = Wav2MFCC(**asdict(self.config))

    @property
    def frame_shift(self) -> Seconds:
        return self.config.frame_shift

    def feature_dim(self, sampling_rate: int) -> int:
        return self.config.num_ceps

    def extract(self, samples: np.ndarray, sampling_rate: int) -> np.ndarray:
        assert sampling_rate == self.config.sampling_rate, f"KaldiFbank was instantiated for sampling_rate " \
                                                           f"{self.config.sampling_rate}, but " \
                                                           f"sampling_rate={sampling_rate} was passed to extract()."
        return self.extractor(torch.from_numpy(samples))[0].numpy()
