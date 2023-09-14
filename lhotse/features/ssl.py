from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch

from lhotse.features.base import FeatureExtractor, register_extractor
from lhotse.utils import Seconds, compute_num_frames_from_samples, is_module_available


@dataclass
class S3PRLSSLConfig:
    """
    In general, the output feature dimension of base model (e.g., wav2vec2) and
    large model (e.g., wav2vec2_large_ll60k) are 768 and 1024, repectively. The
    frame shift (stride) is 0.02s (20ms).

    Please check
        https://github.com/s3prl/s3prl/blob/main/s3prl/upstream/README.md and
        https://s3prl.github.io/s3prl/tutorial/upstream_collection.html
    for details of available self-supervised models.
    """

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

    @property
    def sampling_rate(self) -> int:
        return self.config.sampling_rate

    def feature_dim(self, sampling_rate: int) -> int:
        assert (
            sampling_rate == 16000
        ), f"All the upstream models in S3PRL now only support 16 kHz audio."
        return self.config.feature_dim

    def fix_off_by_one_error(self, feats: np.ndarray, num_samples: int) -> np.ndarray:
        # The last frame is usually shorter than the others.
        # We pad it with zeros to make it the same length as others.
        num_frames, num_features = feats.shape
        expected_num_frames = compute_num_frames_from_samples(
            num_samples=num_samples,
            frame_shift=self.frame_shift,
            sampling_rate=self.sampling_rate,
        )
        num_frames_diff = abs(expected_num_frames - num_frames)
        assert num_frames_diff <= 1
        if num_frames_diff == 1:
            pad = np.zeros([1, num_features])
            feats = np.concatenate([feats, pad], axis=0)
        return feats

    def extract_batch(
        self,
        samples: Union[
            np.ndarray, torch.Tensor, Sequence[np.ndarray], Sequence[torch.Tensor]
        ],
        sampling_rate: int,
        lengths: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]]:
        # S3PRL expects a list of 1D torch tensors.
        if lengths is not None:
            samples = [x[:l] for x, l in zip(samples, lengths)]
        return self.extract(samples=samples, sampling_rate=sampling_rate)

    def extract(
        self,
        samples: Union[
            np.ndarray, torch.Tensor, Sequence[np.ndarray], Sequence[torch.Tensor]
        ],
        sampling_rate: int,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        assert (
            sampling_rate == 16000
        ), f"All the upstream models in S3PRL now only support 16 kHz audio."

        # s3prl expects a batch of 1D torch tensors.
        # Regardless of input type, we return a numpy array or list of numpy arrays.

        input_is_list = isinstance(samples, list)
        if input_is_list:
            samples = [s.squeeze() for s in samples]
        else:
            samples = samples.squeeze()

        # Convert input to a list of 1D torch tensors.
        if input_is_list or samples.ndim > 1:
            samples = [
                torch.from_numpy(s) if isinstance(s, np.ndarray) else s for s in samples
            ]
        else:
            # The user passed a single array/tensor of shape (num_samples,)
            samples = [
                torch.from_numpy(samples)
                if isinstance(samples, np.ndarray)
                else samples
            ]

        samples = [s.to(self.config.device) for s in samples]
        lengths = [s.shape[0] for s in samples]

        self.ssl_model.eval()
        with torch.no_grad():
            feats = self.ssl_model(samples)["hidden_states"][self.config.layer]
        feats = feats.squeeze()

        if feats.ndim == 2:
            # The user passed a single array/tensor of shape (num_samples,)
            feats = feats.cpu().numpy()
            feats = self.fix_off_by_one_error(feats, lengths[0])
            if input_is_list:
                feats = [feats]
        else:
            # The user passed a batch of arrays/tensors of shape (num_samples,)
            # Convert the padded sequence to a list of 1D torch tensors.
            out_lens = [
                compute_num_frames_from_samples(
                    num_samples, self.config.frame_shift, self.config.sampling_rate
                )
                for num_samples in lengths
            ]
            feats = [f[:l].cpu().numpy() for f, l in zip(feats, out_lens)]
            feats = [self.fix_off_by_one_error(f, l) for f, l in zip(feats, lengths)]

            # If all items are of the same shape, stack them into a single array.
            if all(item.shape == feats[0].shape for item in feats[1:]):
                feats = np.stack(feats, axis=0)

        return feats
