from dataclasses import asdict, dataclass
from typing import Any, Dict

import numpy as np

from lhotse.features.base import TorchaudioFeatureExtractor, register_extractor
from lhotse.utils import EPSILON, Seconds


@dataclass
class TorchaudioFbankConfig:
    # Spectogram-related part
    dither: float = 0.0
    window_type: str = "povey"
    # Note that frame_length and frame_shift will be converted to milliseconds before torchaudio/Kaldi sees them
    frame_length: Seconds = 0.025
    frame_shift: Seconds = 0.01
    remove_dc_offset: bool = True
    round_to_power_of_two: bool = True
    energy_floor: float = EPSILON
    min_duration: float = 0.0
    preemphasis_coefficient: float = 0.97
    raw_energy: bool = True

    # Fbank-related part
    low_freq: float = 20.0
    high_freq: float = -400.0
    num_mel_bins: int = 80
    use_energy: bool = False
    vtln_low: float = 100.0
    vtln_high: float = -500.0
    vtln_warp: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "TorchaudioFbankConfig":
        return TorchaudioFbankConfig(**data)


@register_extractor
class TorchaudioFbank(TorchaudioFeatureExtractor):
    """Log Mel energy filter bank feature extractor based on ``torchaudio.compliance.kaldi.fbank`` function."""

    name = "fbank"
    config_type = TorchaudioFbankConfig

    def _feature_fn(self, *args, **kwargs):
        from torchaudio.compliance.kaldi import fbank

        return fbank(*args, **kwargs)

    def feature_dim(self, sampling_rate: int) -> int:
        return self.config.num_mel_bins

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
