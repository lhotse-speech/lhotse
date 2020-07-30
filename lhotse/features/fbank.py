import os
from dataclasses import dataclass

import numpy as np

# Workaround for SoundFile (torchaudio dep) raising exception when a native library, libsndfile1, is not installed.
# Read-the-docs does not allow to modify the Docker containers used to build documentation...
if not os.environ.get('READTHEDOCS', False):
    import torchaudio

from lhotse.features.base import TorchaudioFeatureExtractor, register_extractor
from lhotse.utils import Seconds


@dataclass
class FbankConfig:
    # Spectogram-related part
    dither: float = 0.0
    window_type: str = "povey"
    # Note that frame_length and frame_shift will be converted to milliseconds before torchaudio/Kaldi sees them
    frame_length: Seconds = 0.025
    frame_shift: Seconds = 0.01
    remove_dc_offset: bool = True
    round_to_power_of_two: bool = True
    energy_floor: float = 0.1
    min_duration: float = 0.0
    preemphasis_coefficient: float = 0.97
    raw_energy: bool = True

    # Fbank-related part
    low_freq: float = 20.0
    high_freq: float = 0.0
    num_mel_bins: int = 23
    use_energy: bool = False
    vtln_low: float = 100.0
    vtln_high: float = -500.0
    vtln_warp: float = 1.0


@register_extractor
class Fbank(TorchaudioFeatureExtractor):
    name = 'fbank'
    config_type = FbankConfig
    feature_fn = staticmethod(torchaudio.compliance.kaldi.fbank)

    @staticmethod
    def mix(features_a: np.ndarray, features_b: np.ndarray, gain_b: float) -> np.ndarray:
        return np.log(np.exp(features_a) + gain_b * np.exp(features_b))

    @staticmethod
    def compute_energy(features: np.ndarray) -> float:
        return float(np.sum(np.exp(features)))
