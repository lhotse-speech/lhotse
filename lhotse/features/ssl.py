from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

from lhotse.features.base import FeatureExtractor, register_extractor
from lhotse.utils import (
    EPSILON,
    LOG_EPSILON,
    Seconds,
    compute_num_frames,
    is_module_available,
)


@dataclass
class S3PRLSSLConfig:
    sampling_rate: int = 16000
    ssl_model: str = "wav2vec2_large_ll60k"
    layer: int = -1
    frame_shift: float = 0.02
    feature_dim: int = 1024
    device: str = "cpu"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "S3PRLSSLConfig":
        return S3PRLSSLConfig(**data)


@register_extractor
class S3PRLSSL(FeatureExtractor):
    name = "s3prl-ssl"
    config_type = S3PRLSSLConfig

    def __init__(self, config: Optional[Any] = None):
        super().__init__(config)
        assert is_module_available(
            "s3prl.hub"
        ), "To use s3prl ssl extractors, please install s3prl first."
        import s3prl.hub as hub

        assert self.config.ssl_model in dir(
            hub
        ), f"S3PRL dose not suport model: {self.config.ssl_model}."
        assert (
            self.config.sampling_rate == 16000
        ), f"All the upstream models in S3PRL now only support 16 kHz audio."

        ssl_model = getattr(hub, self.config.ssl_model)()
        self.ssl_model = ssl_model.to(self.config.device)

    @property
    def frame_shift(self) -> Seconds:
        return self.config.frame_shift

    def feature_dim(self, sampling_rate: int) -> int:
        assert (
            sampling_rate == 16000
        ), f"All the upstream models in S3PRL now only support 16 kHz audio."
        return self.config.feature_dim

    def extract(self, samples: np.ndarray, sampling_rate: int) -> np.ndarray:
        assert (
            sampling_rate == 16000
        ), f"All the upstream models in S3PRL now only support 16 kHz audio."

        samples = torch.from_numpy(samples).to(self.config.device)

        self.ssl_model.eval()
        with torch.no_grad():
            feats = self.ssl_model(samples)["hidden_states"][self.config.layer]
        feats = feats.squeeze()

        num_frames, num_features = feats.shape
        duration = round(samples.shape[1] / sampling_rate, ndigits=8)
        expected_num_frames = compute_num_frames(
            duration=duration,
            frame_shift=self.frame_shift,
            sampling_rate=sampling_rate,
        )
        num_frames_diff = expected_num_frames - num_frames
        assert num_frames_diff <= 1
        if num_frames_diff == 1:
            pad = torch.zeros([1, num_features]).to(self.config.device)
            feats = torch.concat([feats, pad], dim=0)

        return feats.cpu().numpy()
