from dataclasses import asdict, dataclass
from typing import Any, Dict

import numpy as np

from lhotse.features.base import TorchaudioFeatureExtractor, register_extractor
from lhotse.utils import EPSILON, Seconds


@dataclass
class SpectrogramConfig:
    # Note that `snip_edges` parameter is missing from config: in order to simplify the relationship between
    #  the duration and the number of frames, we are always setting `snip_edges` to False.
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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "SpectrogramConfig":
        return SpectrogramConfig(**data)


@register_extractor
class Spectrogram(TorchaudioFeatureExtractor):
    """Log spectrogram feature extractor based on ``torchaudio.compliance.kaldi.spectrogram`` function."""

    name = "spectrogram"
    config_type = SpectrogramConfig

    def _feature_fn(self, *args, **kwargs):
        from torchaudio.compliance.kaldi import spectrogram

        return spectrogram(*args, **kwargs)

    def feature_dim(self, sampling_rate: int) -> int:
        from torchaudio.compliance.kaldi import _next_power_of_2

        window_size = int(self.config.frame_length * sampling_rate)
        return (
            _next_power_of_2(window_size) // 2 + 1
            if self.config.round_to_power_of_two
            else window_size
        )

    @staticmethod
    def mix(
        features_a: np.ndarray, features_b: np.ndarray, energy_scaling_factor_b: float
    ) -> np.ndarray:
        # Torchaudio returns log-power spectrum, hence the need for logsumexp
        return np.log(
            np.maximum(
                # protection against log(0); max with EPSILON is adequate since these are energies (always >= 0)
                EPSILON,
                np.exp(features_a) + energy_scaling_factor_b * np.exp(features_b),
            )
        )

    @staticmethod
    def compute_energy(features: np.ndarray) -> float:
        # Torchaudio returns log-power spectrum, hence the need for exp before the sum
        return float(np.sum(np.exp(features)))
