from abc import ABC
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch

from lhotse.features.base import FeatureExtractor, register_extractor
from lhotse.utils import EPSILON, Seconds, is_module_available


@dataclass
class KaldifeatFrameOptions:
    sampling_rate: int = 16000
    frame_shift: Seconds = 0.01
    frame_length: Seconds = 0.025
    dither: float = 0.0  # default was 1.0
    preemph_coeff: float = 0.97
    remove_dc_offset: bool = True
    window_type: str = "povey"
    round_to_power_of_two: bool = True
    blackman_coeff: float = 0.42
    snip_edges: bool = False  # default was True (won't work with Lhotse)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["samp_freq"] = float(d.pop("sampling_rate"))
        d["frame_shift_ms"] = d.pop("frame_shift") * 1000.0
        d["frame_length_ms"] = d.pop("frame_length") * 1000.0
        return d

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "KaldifeatFrameOptions":
        data = data.copy()
        if "samp_freq" in data:
            data["sampling_rate"] = int(data.pop("samp_freq"))
        for key in ["frame_shift_ms", "frame_length_ms"]:
            if key in data:
                data[key.replace("_ms", "")] = data.pop(key) / 1000
        return KaldifeatFrameOptions(**data)


@dataclass
class KaldifeatMelOptions:
    num_bins: int = 80  # default was 23
    low_freq: float = 20.0
    high_freq: float = -400.0  # default was 0.0
    vtln_low: float = 100.0
    vtln_high: float = -500.0
    debug_mel: bool = False
    htk_mode: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "KaldifeatMelOptions":
        return KaldifeatMelOptions(**data)


class KaldifeatExtractor(FeatureExtractor, ABC):
    """
    Base class with shared implementation for kaldifeat feature extractors.
    Derived classes are expected to set ``self.extractor`` inside ``__init__``.
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        super().__init__(config=config)
        assert is_module_available(
            "kaldifeat"
        ), 'To use kaldifeat extractors, please "pip install kaldifeat" first.'

    @property
    def device(self) -> Union[str, torch.device]:
        return self.config.device

    def extract_batch(
        self,
        samples: Union[
            np.ndarray, torch.Tensor, Sequence[np.ndarray], Sequence[torch.Tensor]
        ],
        sampling_rate: int,
    ) -> Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]]:
        return self.extract(samples=samples, sampling_rate=sampling_rate)

    def extract(
        self,
        samples: Union[
            np.ndarray, torch.Tensor, Sequence[np.ndarray], Sequence[torch.Tensor]
        ],
        sampling_rate: int,
    ) -> Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]]:
        # Check for sampling rate compatibility.
        expected_sr = self.config.frame_opts.sampling_rate
        assert sampling_rate == expected_sr, (
            f"Mismatched sampling rate: extractor expects {expected_sr}, "
            f"got {sampling_rate}"
        )

        # kaldifeat expects a list of 1D torch tensors.
        # If we got a torch tensor / list of torch tensors in the input,
        # we'll also return torch tensors. If we got numpy arrays, we
        # will convert back to numpy.
        maybe_as_numpy = lambda x: x
        input_is_list = False

        if isinstance(samples, list):
            input_is_list = True
            pass  # nothing to do with `samples`
        elif samples.ndim > 1:
            samples = list(samples)
        else:
            # The user passed an array/tensor of shape (num_samples,)
            samples = [samples]

        for idx in range(len(samples)):
            if isinstance(samples[idx], np.ndarray):
                samples[idx] = torch.from_numpy(samples[idx])
                maybe_as_numpy = lambda x: x.numpy()
            if samples[idx].ndim == 2:
                # ndim could be > 1 if the input is a list of arrays/tensors.
                samples[idx] = samples[idx].squeeze()

        # Actual feature extraction.
        result = self.extractor(samples, chunk_size=self.config.chunk_size)

        # If all items are of the same shape, concatenate
        if len(result) == 1:
            if input_is_list:
                return [maybe_as_numpy(result[0])]
            else:
                return maybe_as_numpy(result[0])
        elif all(item.shape == result[0].shape for item in result[1:]):
            return maybe_as_numpy(torch.stack(result, dim=0))
        else:
            return [maybe_as_numpy(r) for r in result]

    @property
    def frame_shift(self) -> Seconds:
        return self.config.frame_opts.frame_shift


@dataclass
class KaldifeatFbankConfig:
    frame_opts: KaldifeatFrameOptions = KaldifeatFrameOptions()
    mel_opts: KaldifeatMelOptions = KaldifeatMelOptions()
    use_energy: bool = False
    energy_floor: float = EPSILON  # default was 0.0
    raw_energy: bool = True
    htk_compat: bool = False
    use_log_fbank: bool = True
    use_power: bool = True
    device: Union[str, torch.device] = "cpu"

    # This is an extra setting compared to kaldifeat FbankOptions:
    # by default, we'll ask kaldifeat to compute the feats in chunks
    # to avoid excessive memory usage.
    chunk_size: Optional[int] = 100 * 60 * 20  # 20 minutes (assuming 100 frame/s)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["frame_opts"] = self.frame_opts.to_dict()
        d["mel_opts"] = self.mel_opts.to_dict()
        return d

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "KaldifeatFbankConfig":
        frame_opts = KaldifeatFrameOptions.from_dict(data.pop("frame_opts"))
        mel_opts = KaldifeatMelOptions.from_dict(data.pop("mel_opts"))
        return KaldifeatFbankConfig(frame_opts=frame_opts, mel_opts=mel_opts, **data)


@register_extractor
class KaldifeatFbank(KaldifeatExtractor):
    """Log Mel energy filter bank feature extractor based on ``kaldifeat`` package."""

    name = "kaldifeat-fbank"
    config_type = KaldifeatFbankConfig

    def __init__(self, config: Optional[KaldifeatFbankConfig] = None) -> None:
        super().__init__(config)
        import kaldifeat

        self.extractor = kaldifeat.Fbank(
            kaldifeat.FbankOptions.from_dict(self.config.to_dict())
        )

    def feature_dim(self, sampling_rate: int) -> int:
        return self.config.mel_opts.num_bins

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
class KaldifeatMfccConfig:
    frame_opts: KaldifeatFrameOptions = KaldifeatFrameOptions()
    mel_opts: KaldifeatMelOptions = KaldifeatMelOptions(num_bins=23)
    num_ceps: int = 13
    use_energy: bool = False
    energy_floor: float = EPSILON  # default was 0.0
    raw_energy: bool = True
    cepstral_lifter: float = 22.0
    htk_compat: bool = False
    device: Union[str, torch.device] = "cpu"

    # This is an extra setting compared to kaldifeat FbankOptions:
    # by default, we'll ask kaldifeat to compute the feats in chunks
    # to avoid excessive memory usage.
    chunk_size: Optional[int] = 1000

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["frame_opts"] = self.frame_opts.to_dict()
        d["mel_opts"] = self.mel_opts.to_dict()
        return d

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "KaldifeatMfccConfig":
        frame_opts = KaldifeatFrameOptions.from_dict(data.pop("frame_opts"))
        mel_opts = KaldifeatMelOptions.from_dict(data.pop("mel_opts"))
        return KaldifeatMfccConfig(frame_opts=frame_opts, mel_opts=mel_opts, **data)


@register_extractor
class KaldifeatMfcc(KaldifeatExtractor):
    """MFCC feature extractor based on ``kaldifeat`` package."""

    name = "kaldifeat-mfcc"
    config_type = KaldifeatMfccConfig

    def __init__(self, config: Optional[KaldifeatMfccConfig] = None) -> None:
        super().__init__(config)
        import kaldifeat

        self.extractor = kaldifeat.Mfcc(
            kaldifeat.MfccOptions.from_dict(self.config.to_dict())
        )

    def feature_dim(self, sampling_rate: int) -> int:
        return self.config.num_ceps
