from dataclasses import asdict, dataclass
from typing import Any, Dict

import numpy as np
import opensmile

from lhotse.features.base import FeatureExtractor, register_extractor
from lhotse.utils import Seconds


@dataclass
class OpenSmileConfig:
    # Note that `snip_edges` parameter is missing from config: in order to simplify the relationship between
    #  the duration and the number of frames, we are always setting `snip_edges` to False.
  
    feature_set: opensmile.FeatureSet = opensmile.FeatureSet.ComParE_2016
    feature_level: opensmile.FeatureLevel = opensmile.FeatureLevel.LowLevelDescriptors
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "OpenSmileConfig":
        return OpenSmileConfig(**data)


@register_extractor
class OpenSmileWrapper(FeatureExtractor):
    """Wrapper for extraction of features implemented in OpenSmile."""

    name = "opensmile-wrapper"
    config_type = OpenSmileConfig     
    
    @property
    def frame_shift(self) -> Seconds:
        return 0

    def feature_dim(self, sampling_rate: int) -> int:
        return 0

    def extract(self, samples: np.ndarray, sampling_rate: int) -> np.ndarray:
 #       if is_module_available("opensmile"):
 #           import opensmile
 #       else:
 #           raise ImportError(
 #               "OpenSmile is not installed. Please install opensmile before using OpenSmileWrapper."
 #           )
        extractor = opensmile.Smile(
                    feature_set=self.config.feature_set,
                    feature_level=self.config.feature_level,
                    sampling_rate=sampling_rate,
                    num_workers=1,
                    )
        return extractor.process_signal(samples, sampling_rate=sampling_rate).to_numpy()    
    

#    @staticmethod
#    def mix(
#        features_a: np.ndarray, features_b: np.ndarray, energy_scaling_factor_b: float
#    ) -> np.ndarray:
#        # Torchaudio returns log-power spectrum, hence the need for logsumexp
#        return np.log(
#            np.maximum(
#                # protection against log(0); max with EPSILON is adequate since these are energies (always >= 0)
#                EPSILON,
#                np.exp(features_a) + energy_scaling_factor_b * np.exp(features_b),
#            )
#        )
#
#    @staticmethod
#    def compute_energy(features: np.ndarray) -> float:
#        # Torchaudio returns log-power spectrum, hence the need for exp before the sum
#        return float(np.sum(np.exp(features)))
