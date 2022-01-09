from dataclasses import asdict, dataclass
from typing import Any, Dict

from lhotse.features.base import TorchaudioFeatureExtractor, register_extractor
from lhotse.utils import EPSILON, Seconds


@dataclass
class TorchaudioMfccConfig:
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

    # MFCC-related part
    low_freq: float = 20.0
    high_freq: float = -400.0
    num_mel_bins: int = 23
    use_energy: bool = False
    vtln_low: float = 100.0
    vtln_high: float = -500.0
    vtln_warp: float = 1.0
    cepstral_lifter: float = 22.0
    num_ceps: int = 13

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "TorchaudioMfccConfig":
        return TorchaudioMfccConfig(**data)


@register_extractor
class TorchaudioMfcc(TorchaudioFeatureExtractor):
    """MFCC feature extractor based on ``torchaudio.compliance.kaldi.mfcc`` function."""

    name = "mfcc"
    config_type = TorchaudioMfccConfig

    def _feature_fn(self, *args, **kwargs):
        from torchaudio.compliance.kaldi import mfcc

        return mfcc(*args, **kwargs)

    def feature_dim(self, sampling_rate: int) -> int:
        return self.config.num_ceps
