from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from lhotse.features.base import FeatureExtractor, register_extractor
from lhotse.utils import (
    EPSILON,
    Seconds,
    asdict_nonull,
    compute_num_frames_from_samples,
    is_module_available,
)


def log_mel_spectrogram(
    audio: Union[np.ndarray, torch.Tensor],
    filters: torch.Tensor,
    n_mels: int = 80,
    n_fft: int = 400,
    window: torch.Tensor = None,
    hop_length: int = 160,
    sampling_rate: int = 16000,
    device: Optional[Union[str, torch.device]] = None,
):
    """
    From https://github.com/openai/whisper/blob/main/whisper/audio.py

    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 is supported

    device: Optional[Union[str, torch.device]]
        If given, the audio tensor is moved to this device before STFT

    Returns
    -------
    torch.Tensor, shape = (n_frames, 80)
        A Tensor that contains the Mel spectrogram
    """
    if not torch.is_tensor(audio):
        audio = torch.from_numpy(audio)

    if device is not None:
        audio = audio.to(device)

    if len(audio.shape) == 2:
        if audio.shape[0] > 1:
            raise ValueError("Whisper Fbank works only with single-channel recordings.")
        else:
            audio = audio[0]
    assert (
        len(audio.shape) == 1
    ), f"Whisper Fbank works only with single-channel recordings (shape: {audio.shape})"

    stft = torch.stft(audio, n_fft, hop_length, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    padding = compute_num_frames_from_samples(
        num_samples=len(audio),
        frame_shift=hop_length / sampling_rate,
        sampling_rate=sampling_rate,
    )
    if padding > log_spec.shape[1]:
        log_spec = torch.nn.functional.pad(
            log_spec, (0, padding - log_spec.shape[1]), mode="constant"
        )
    # change shape from 80, n_frames to n_frames,80
    log_spec = log_spec.transpose(0, 1)

    return log_spec


@dataclass
class WhisperFbankConfig:
    num_filters: int = 80
    device: str = "cpu"

    def to_dict(self) -> Dict[str, Any]:
        return asdict_nonull(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "WhisperFbankConfig":
        return WhisperFbankConfig(**data)


@register_extractor
class WhisperFbank(FeatureExtractor):
    name = "whisper-fbank"
    config_type = WhisperFbankConfig

    def __init__(self, config: Optional[WhisperFbankConfig] = None):
        super().__init__(config=config)
        self.sampling_rate = 16000
        self.hop_length = 160
        self.n_fft = 400
        self.num_filters = self.config.num_filters
        if is_module_available("librosa"):
            import librosa
        else:
            raise ImportError(
                "Librosa is not installed. Please install librosa before using LibrosaFbank extractor."
            )
        window = torch.hann_window(self.n_fft).to(self.config.device)
        filters = librosa.filters.mel(
            sr=self.sampling_rate, n_fft=self.n_fft, n_mels=self.num_filters
        )
        self.filters = torch.from_numpy(filters).to(self.config.device)
        self.window = torch.hann_window(self.n_fft).to(self.config.device)

    @property
    def device(self) -> Union[str, torch.device]:
        return self.config.device

    @property
    def frame_shift(self) -> Seconds:
        return self.hop_length / self.sampling_rate

    def to(self, device: str):
        self.config.device = device

    def feature_dim(self, sampling_rate: int) -> int:
        return self.num_filters

    def extract(
        self, samples: Union[np.ndarray, torch.Tensor], sampling_rate: int
    ) -> Union[np.ndarray, torch.Tensor]:
        assert sampling_rate == self.sampling_rate, (
            f"Fbank was instantiated for sampling_rate "
            f"{self.sampling_rate}, but "
            f"sampling_rate={sampling_rate} was passed to extract(). "
            "Note you can use CutSet/RecordingSet.resample() to change the audio sampling rate."
        )

        is_numpy = False
        if not isinstance(samples, torch.Tensor):
            samples = torch.from_numpy(samples)
            is_numpy = True

        feats = log_mel_spectrogram(
            samples,
            filters=self.filters,
            n_fft=self.n_fft,
            window=self.window,
            n_mels=self.num_filters,
            device=self.device,
        )

        if is_numpy:
            return feats.cpu().numpy()
        else:
            return feats

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
