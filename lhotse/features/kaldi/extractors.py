import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch

from lhotse.features.base import FeatureExtractor, register_extractor
from lhotse.features.kaldi.layers import (
    Wav2LogFilterBank,
    Wav2LogSpec,
    Wav2MFCC,
    Wav2Spec,
)
from lhotse.utils import (
    EPSILON,
    Seconds,
    asdict_nonull,
    compute_num_frames_from_samples,
)


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
    torchaudio_compatible_mel_scale: bool = True
    device: str = "cpu"

    def __post_init__(self):
        # This is to help users transition to a different Fbank implementation
        # from torchaudio.compliance.kaldi.fbank(), where the arg had a different name.
        if self.num_mel_bins is not None:
            self.num_filters = self.num_mel_bins
            self.num_mel_bins = None

        if self.snip_edges:
            warnings.warn(
                "`snip_edges` is set to True, which may cause issues in duration to num-frames conversion in Lhotse."
            )

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
        config_dict = self.config.to_dict()
        config_dict.pop("device")
        self.extractor = Wav2LogFilterBank(**config_dict).to(self.device).eval()

    @property
    def device(self) -> Union[str, torch.device]:
        return self.config.device

    @property
    def frame_shift(self) -> Seconds:
        return self.config.frame_shift

    def to(self, device: str):
        self.config.device = device
        self.extractor.to(device)

    def feature_dim(self, sampling_rate: int) -> int:
        return self.config.num_filters

    def extract(
        self, samples: Union[np.ndarray, torch.Tensor], sampling_rate: int
    ) -> Union[np.ndarray, torch.Tensor]:
        assert sampling_rate == self.config.sampling_rate, (
            f"Fbank was instantiated for sampling_rate "
            f"{self.config.sampling_rate}, but "
            f"sampling_rate={sampling_rate} was passed to extract(). "
            "Note you can use CutSet/RecordingSet.resample() to change the audio sampling rate."
        )

        is_numpy = False
        if not isinstance(samples, torch.Tensor):
            samples = torch.from_numpy(samples)
            is_numpy = True

        if samples.ndim == 1:
            samples = samples.unsqueeze(0)

        feats = self.extractor(samples.to(self.device))[0]

        if is_numpy:
            return feats.cpu().numpy()
        else:
            return feats

    def extract_batch(
        self,
        samples: Union[
            np.ndarray, torch.Tensor, Sequence[np.ndarray], Sequence[torch.Tensor]
        ],
        sampling_rate: int,
        lengths: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]]:
        return _extract_batch(
            self.extractor,
            samples,
            sampling_rate,
            frame_shift=self.frame_shift,
            lengths=lengths,
            device=self.device,
        )

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
    torchaudio_compatible_mel_scale: bool = True
    num_mel_bins: Optional[int] = None  # do not use
    norm_filters: bool = False
    num_ceps: int = 13
    cepstral_lifter: int = 22
    device: str = "cpu"

    def __post_init__(self):
        # This is to help users transition to a different Mfcc implementation
        # from torchaudio.compliance.kaldi.fbank(), where the arg had a different name.
        if self.num_mel_bins is not None:
            self.num_filters = self.num_mel_bins
            self.num_mel_bins = None

        if self.snip_edges:
            warnings.warn(
                "`snip_edges` is set to True, which may cause issues in duration to num-frames conversion in Lhotse."
            )

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
        config_dict = self.config.to_dict()
        config_dict.pop("device")
        self.extractor = Wav2MFCC(**config_dict).to(self.device).eval()

    @property
    def device(self) -> Union[str, torch.device]:
        return self.config.device

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
            f"sampling_rate={sampling_rate} was passed to extract(). "
            "Note you can use CutSet/RecordingSet.resample() to change the audio sampling rate."
        )

        is_numpy = False
        if not isinstance(samples, torch.Tensor):
            samples = torch.from_numpy(samples)
            is_numpy = True

        if samples.ndim == 1:
            samples = samples.unsqueeze(0)

        feats = self.extractor(samples.to(self.device))[0]

        if is_numpy:
            return feats.cpu().numpy()
        else:
            return feats

    def extract_batch(
        self,
        samples: Union[
            np.ndarray, torch.Tensor, Sequence[np.ndarray], Sequence[torch.Tensor]
        ],
        sampling_rate: int,
        lengths: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]]:
        return _extract_batch(
            self.extractor,
            samples,
            sampling_rate,
            frame_shift=self.frame_shift,
            lengths=lengths,
            device=self.device,
        )


@dataclass
class SpectrogramConfig:
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
    device: str = "cpu"

    def __post_init__(self):
        if self.snip_edges:
            warnings.warn(
                "`snip_edges` is set to True, which may cause issues in duration to num-frames conversion in Lhotse."
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict_nonull(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "SpectrogramConfig":
        return SpectrogramConfig(**data)


@register_extractor
class Spectrogram(FeatureExtractor):
    name = "kaldi-spectrogram"
    config_type = SpectrogramConfig

    def __init__(self, config: Optional[SpectrogramConfig] = None):
        super().__init__(config=config)
        config_dict = self.config.to_dict()
        config_dict.pop("device")
        self.extractor = Wav2Spec(**config_dict).to(self.device).eval()

    @property
    def device(self) -> Union[str, torch.device]:
        return self.config.device

    @property
    def frame_shift(self) -> Seconds:
        return self.config.frame_shift

    def feature_dim(self, sampling_rate: int) -> int:
        return self.extractor.fft_length // 2 + 1

    def extract(
        self, samples: Union[np.ndarray, torch.Tensor], sampling_rate: int
    ) -> Union[np.ndarray, torch.Tensor]:
        assert sampling_rate == self.config.sampling_rate, (
            f"Spectrogram was instantiated for sampling_rate "
            f"{self.config.sampling_rate}, but "
            f"sampling_rate={sampling_rate} was passed to extract(). "
            "Note you can use CutSet/RecordingSet.resample() to change the audio sampling rate."
        )

        is_numpy = False
        if not isinstance(samples, torch.Tensor):
            samples = torch.from_numpy(samples)
            is_numpy = True

        if samples.ndim == 1:
            samples = samples.unsqueeze(0)

        feats = self.extractor(samples.to(self.device))[0]

        if is_numpy:
            return feats.cpu().numpy()
        else:
            return feats.cpu()

    def extract_batch(
        self,
        samples: Union[
            np.ndarray, torch.Tensor, Sequence[np.ndarray], Sequence[torch.Tensor]
        ],
        sampling_rate: int,
        lengths: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]]:
        return _extract_batch(
            self.extractor,
            samples,
            sampling_rate,
            frame_shift=self.frame_shift,
            lengths=lengths,
            device=self.device,
        )

    @staticmethod
    def mix(
        features_a: np.ndarray, features_b: np.ndarray, energy_scaling_factor_b: float
    ) -> np.ndarray:
        return features_a + energy_scaling_factor_b * features_b

    @staticmethod
    def compute_energy(features: np.ndarray) -> float:
        return float(np.sum(features))


@dataclass
class LogSpectrogramConfig:
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
    device: str = "cpu"

    def __post_init__(self):
        if self.snip_edges:
            warnings.warn(
                "`snip_edges` is set to True, which may cause issues in duration to num-frames conversion in Lhotse."
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict_nonull(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "LogSpectrogramConfig":
        return LogSpectrogramConfig(**data)


@register_extractor
class LogSpectrogram(FeatureExtractor):
    name = "kaldi-log-spectrogram"
    config_type = LogSpectrogramConfig

    def __init__(self, config: Optional[LogSpectrogramConfig] = None):
        super().__init__(config=config)
        config_dict = self.config.to_dict()
        config_dict.pop("device")
        self.extractor = Wav2LogSpec(**config_dict).to(self.device).eval()

    @property
    def device(self) -> Union[str, torch.device]:
        return self.config.device

    @property
    def frame_shift(self) -> Seconds:
        return self.config.frame_shift

    def feature_dim(self, sampling_rate: int) -> int:
        return self.extractor.fft_length // 2 + 1

    def extract(
        self, samples: Union[np.ndarray, torch.Tensor], sampling_rate: int
    ) -> Union[np.ndarray, torch.Tensor]:
        assert sampling_rate == self.config.sampling_rate, (
            f"Spectrogram was instantiated for sampling_rate "
            f"{self.config.sampling_rate}, but "
            f"sampling_rate={sampling_rate} was passed to extract(). "
            "Note you can use CutSet/RecordingSet.resample() to change the audio sampling rate."
        )

        is_numpy = False
        if not isinstance(samples, torch.Tensor):
            samples = torch.from_numpy(samples)
            is_numpy = True

        if samples.ndim == 1:
            samples = samples.unsqueeze(0)

        feats = self.extractor(samples.to(self.device))[0]

        if is_numpy:
            return feats.cpu().numpy()
        else:
            return feats.cpu()

    def extract_batch(
        self,
        samples: Union[
            np.ndarray, torch.Tensor, Sequence[np.ndarray], Sequence[torch.Tensor]
        ],
        sampling_rate: int,
        lengths: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]]:
        return _extract_batch(
            self.extractor,
            samples,
            sampling_rate,
            frame_shift=self.frame_shift,
            lengths=lengths,
            device=self.device,
        )

    @staticmethod
    def mix(
        features_a: np.ndarray, features_b: np.ndarray, energy_scaling_factor_b: float
    ) -> np.ndarray:
        return features_a + energy_scaling_factor_b * features_b

    @staticmethod
    def compute_energy(features: np.ndarray) -> float:
        return float(np.sum(features))


def _extract_batch(
    extractor: FeatureExtractor,
    samples: Union[
        np.ndarray, torch.Tensor, Sequence[np.ndarray], Sequence[torch.Tensor]
    ],
    sampling_rate: int,
    frame_shift: Seconds = 0.01,
    lengths: Optional[Union[np.ndarray, torch.Tensor]] = None,
    device: Union[str, torch.device] = "cpu",
) -> Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]]:
    input_is_list = False
    input_is_torch = False

    if lengths is not None:
        feat_lens = [
            compute_num_frames_from_samples(l, frame_shift, sampling_rate)
            for l in lengths
        ]
        assert isinstance(
            samples, torch.Tensor
        ), "If `lengths` is provided, `samples` must be a batched and padded torch.Tensor."
    else:
        if isinstance(samples, list):
            input_is_list = True
            pass  # nothing to do with `samples`
        elif samples.ndim > 1:
            samples = list(samples)
        else:
            # The user passed an array/tensor of shape (num_samples,)
            samples = [samples.reshape(1, -1)]

        if any(isinstance(x, torch.Tensor) for x in samples):
            input_is_torch = True

        samples = [
            torch.from_numpy(x).squeeze() if isinstance(x, np.ndarray) else x.squeeze()
            for x in samples
        ]
        feat_lens = [
            compute_num_frames_from_samples(
                num_samples=len(x),
                frame_shift=extractor.frame_shift,
                sampling_rate=sampling_rate,
            )
            for x in samples
        ]
        samples = torch.nn.utils.rnn.pad_sequence(samples, batch_first=True)

    # Perform feature extraction
    input_device = samples.device
    feats = extractor(samples.to(device))
    feats.to(input_device)
    result = [feats[i, : feat_lens[i]] for i in range(len(samples))]

    if not input_is_torch:
        result = [x.numpy() for x in result]

    # If all items are of the same shape, concatenate
    if len(result) == 1:
        if input_is_list:
            return result
        else:
            return result[0]
    elif all(item.shape == result[0].shape for item in result[1:]):
        if input_is_torch:
            return torch.stack(result, dim=0)
        else:
            return np.stack(result, axis=0)
    else:
        return result
