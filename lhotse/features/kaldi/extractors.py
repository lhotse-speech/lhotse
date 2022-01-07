from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from lhotse.features.base import FeatureExtractor, register_extractor
from lhotse.features.kaldi.layers import Wav2LogFilterBank, Wav2MFCC
from lhotse.utils import EPSILON, Seconds, asdict_nonull


@dataclass
class FbankConfig:
    sampling_rate: int = 16000
    frame_length: Seconds = 0.025
    frame_shift: Seconds = 0.01
    round_to_power_of_two: bool = True
    remove_dc_offset: bool = True
    preemph_coeff: float = 0.97
    window_type: str = "povey"
    dither: float = 0.0
    snip_edges: bool = False
    energy_floor: float = EPSILON
    raw_energy: bool = True
    use_energy: bool = False
    use_fft_mag: bool = False
    low_freq: float = 20.0
    high_freq: float = -400.0
    num_filters: int = 80
    num_mel_bins: Optional[int] = None  # do not use
    norm_filters: bool = False

    def __post_init__(self):
        # This is to help users transition to a different Fbank implementation
        # from torchaudio.compliance.kaldi.fbank(), where the arg had a different name.
        if self.num_mel_bins is not None:
            self.num_filters = self.num_mel_bins
            self.num_mel_bins = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict_nonull(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "FbankConfig":
        return FbankConfig(**data)


@register_extractor
class Fbank(FeatureExtractor):
    name = "kaldi-fbank"
    config_type = FbankConfig

    def __init__(self, config: Optional[FbankConfig] = None):
        super().__init__(config=config)
        self.extractor = Wav2LogFilterBank(**self.config.to_dict()).eval()

    @property
    def frame_shift(self) -> Seconds:
        return self.config.frame_shift

    def feature_dim(self, sampling_rate: int) -> int:
        return self.config.num_filters

    def extract(
        self, samples: Union[np.ndarray, torch.Tensor], sampling_rate: int
    ) -> Union[np.ndarray, torch.Tensor]:
        assert sampling_rate == self.config.sampling_rate, (
            f"Fbank was instantiated for sampling_rate "
            f"{self.config.sampling_rate}, but "
            f"sampling_rate={sampling_rate} was passed to extract()."
        )

        is_numpy = False
        if not isinstance(samples, torch.Tensor):
            samples = torch.from_numpy(samples)
            is_numpy = True

        if samples.ndim == 1:
            samples = samples.unsqueeze(0)

        feats = self.extractor(samples)[0]

        if is_numpy:
            return feats.cpu().numpy()
        else:
            return feats

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
class MfccConfig:
    sampling_rate: int = 16000
    frame_length: Seconds = 0.025
    frame_shift: Seconds = 0.01
    round_to_power_of_two: bool = True
    remove_dc_offset: bool = True
    preemph_coeff: float = 0.97
    window_type: str = "povey"
    dither: float = 0.0
    snip_edges: bool = False
    energy_floor: float = EPSILON
    raw_energy: bool = True
    use_energy: bool = False
    use_fft_mag: bool = False
    low_freq: float = 20.0
    high_freq: float = -400.0
    num_filters: int = 23
    num_mel_bins: Optional[int] = None  # do not use
    norm_filters: bool = False
    num_ceps: int = 13
    cepstral_lifter: int = 22

    def __post_init__(self):
        # This is to help users transition to a different Mfcc implementation
        # from torchaudio.compliance.kaldi.fbank(), where the arg had a different name.
        if self.num_mel_bins is not None:
            self.num_filters = self.num_mel_bins
            self.num_mel_bins = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict_nonull(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "MfccConfig":
        return MfccConfig(**data)


@register_extractor
class Mfcc(FeatureExtractor):
    name = "kaldi-mfcc"
    config_type = MfccConfig

    def __init__(self, config: Optional[MfccConfig] = None):
        super().__init__(config=config)
        self.extractor = Wav2MFCC(**self.config.to_dict()).eval()

    @property
    def frame_shift(self) -> Seconds:
        return self.config.frame_shift

    def feature_dim(self, sampling_rate: int) -> int:
        return self.config.num_ceps

    def extract(
        self, samples: Union[np.ndarray, torch.Tensor], sampling_rate: int
    ) -> Union[np.ndarray, torch.Tensor]:
        assert sampling_rate == self.config.sampling_rate, (
            f"Mfcc was instantiated for sampling_rate "
            f"{self.config.sampling_rate}, but "
            f"sampling_rate={sampling_rate} was passed to extract()."
        )

        is_numpy = False
        if not isinstance(samples, torch.Tensor):
            samples = torch.from_numpy(samples)
            is_numpy = True

        if samples.ndim == 1:
            samples = samples.unsqueeze(0)

        feats = self.extractor(samples)[0]

        if is_numpy:
            return feats.cpu().numpy()
        else:
            return feats
