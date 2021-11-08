from dataclasses import asdict, dataclass
from typing import Any, Dict

import numpy as np

from lhotse.features.base import FeatureExtractor, register_extractor
from lhotse.utils import (
    EPSILON,
    LOG_EPSILON,
    Seconds,
    compute_num_frames,
    is_module_available,
)


@dataclass
class LibrosaFbankConfig:
    """Default librosa config with values consistent with various TTS projects.

    This config is intended for use with popular TTS projects such as [ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN)
    Warning: You may need to normalize your features.
    """

    sampling_rate: int = 22050
    fft_size: int = 1024
    hop_size: int = 256
    win_length: int = None
    window: str = "hann"
    num_mel_bins: int = 80
    fmin: int = 80
    fmax: int = 7600

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "LibrosaFbankConfig":
        return LibrosaFbankConfig(**data)


def pad_or_truncate_features(
    feats: np.ndarray,
    expected_num_frames: int,
    abs_tol: int = 1,
    pad_value: float = LOG_EPSILON,
):
    frames_diff = feats.shape[0] - expected_num_frames

    if 0 < frames_diff <= abs_tol:
        feats = feats[:expected_num_frames]
    elif -abs_tol <= frames_diff < 0:
        feats = np.pad(
            feats,
            ((0, -frames_diff), (0, 0)),
            mode="constant",
            constant_values=LOG_EPSILON,
        )
    elif abs(frames_diff) > abs_tol:
        raise ValueError(
            f"Expected {expected_num_frames} source_feats; feats.shape[0] = {feats.shape[0]}"
        )

    return feats


def logmelfilterbank(
    audio: np.ndarray,
    sampling_rate: int,
    fft_size: int = 1024,
    hop_size: int = 256,
    win_length: int = None,
    window: str = "hann",
    num_mel_bins: int = 80,
    fmin: int = 80,
    fmax: int = 7600,
    eps: float = EPSILON,
):
    """Compute log-Mel filterbank feature.

    Args:
        audio (ndarray): Audio signal (T,).
        sampling_rate (int): Sampling rate.
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length. If set to None, it will be the same as fft_size.
        window (str): Window function type.
        num_mel_bins (int): Number of mel basis.
        fmin (int): Minimum frequency in mel basis calculation.
        fmax (int): Maximum frequency in mel basis calculation.
        eps (float): Epsilon value to avoid inf in log calculation.
    Returns:
        ndarray: Log Mel filterbank feature (#source_feats, num_mel_bins).
    """
    if is_module_available("librosa"):
        import librosa
    else:
        raise ImportError(
            "Librosa is not installed. Please install librosa before using LibrosaFbank extractor."
        )

    if len(audio.shape) == 2:
        assert (
            audio.shape[0] == 1
        ), f"LibrosaFbank works only with single-channel recordings (shape: {audio.shape})"
        audio = audio[0]
    else:
        assert (
            len(audio.shape) == 1
        ), f"LibrosaFbank works only with single-channel recordings (shape: {audio.shape})"

    x_stft = librosa.stft(
        audio,
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=win_length,
        window=window,
        pad_mode="reflect",
    )
    spc = np.abs(x_stft).T

    fmin = 0 if fmin is None else fmin
    fmax = sampling_rate / 2 if fmax is None else fmax
    mel_basis = librosa.filters.mel(sampling_rate, fft_size, num_mel_bins, fmin, fmax)

    feats = np.log10(np.maximum(eps, np.dot(spc, mel_basis.T)))

    expected_num_frames = compute_num_frames(
        duration=len(audio) / sampling_rate,
        frame_shift=hop_size / sampling_rate,
        sampling_rate=sampling_rate,
    )
    feats = pad_or_truncate_features(feats, expected_num_frames)
    return feats


@register_extractor
class LibrosaFbank(FeatureExtractor):
    """Librosa fbank feature extractor

    Differs from Fbank extractor in that it uses librosa backend for stft and mel scale calculations.
    It can be easily configured to be compatible with existing speech-related projects that use librosa features.
    """

    name = "librosa-fbank"
    config_type = LibrosaFbankConfig

    @property
    def frame_shift(self) -> Seconds:
        return self.config.hop_size / self.config.sampling_rate

    def feature_dim(self, sampling_rate: int) -> int:
        return self.config.num_mel_bins

    def extract(self, samples: np.ndarray, sampling_rate: int) -> np.ndarray:
        assert sampling_rate == self.config.sampling_rate
        return logmelfilterbank(samples, **asdict(self.config))

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
