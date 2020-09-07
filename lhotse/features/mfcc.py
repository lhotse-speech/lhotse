from dataclasses import dataclass

import torchaudio

from lhotse.features.base import TorchaudioFeatureExtractor, register_extractor
from lhotse.utils import Seconds


@dataclass
class MfccConfig:
    # Spectogram-related part
    dither: float = 0.0
    window_type: str = "povey"
    # Note that frame_length and frame_shift will be converted to milliseconds before torchaudio/Kaldi sees them
    frame_length: Seconds = 0.025
    frame_shift: Seconds = 0.01
    remove_dc_offset: bool = True
    round_to_power_of_two: bool = True
    energy_floor: float = 1e-10
    min_duration: float = 0.0
    preemphasis_coefficient: float = 0.97
    raw_energy: bool = True

    # MFCC-related part
    low_freq: float = 20.0
    high_freq: float = 0.0
    num_mel_bins: int = 23
    use_energy: bool = False
    vtln_low: float = 100.0
    vtln_high: float = -500.0
    vtln_warp: float = 1.0
    cepstral_lifter: float = 22.0
    num_ceps: int = 13


@register_extractor
class Mfcc(TorchaudioFeatureExtractor):
    """MFCC feature extractor based on ``torchaudio.compliance.kaldi.mfcc`` function."""
    name = 'mfcc'
    config_type = MfccConfig
    feature_fn = staticmethod(torchaudio.compliance.kaldi.mfcc)

    def feature_dim(self, sampling_rate: int) -> int:
        return self.config.num_ceps
