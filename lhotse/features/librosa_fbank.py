from dataclasses import asdict, dataclass

import numpy as np
from lhotse import FeatureExtractor
from lhotse.features.base import register_extractor
from lhotse.utils import Seconds, compute_num_frames, is_module_available


if is_module_available("librosa"):
    import librosa
else:
    raise ImportError(
        "Librosa is not installed. Please install librosa before using LibrosaFbank extractor."
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


def logmelfilterbank(
    audio,
    sampling_rate,
    fft_size=1024,
    hop_size=256,
    win_length=None,
    window="hann",
    num_mel_bins=80,
    fmin=80,
    fmax=7600,
    eps=1e-10,
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
    assert len(audio.shape) == 2
    assert audio.shape[0] == 1
    audio = audio[0]
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
    if feats.shape[0] > expected_num_frames:
        feats = feats[:expected_num_frames, :]
    elif feats.shape[0] < expected_num_frames:
        raise ValueError(
            f"Expected {expected_num_frames} source_feats; feats.shape[0] = {feats.shape[0]}"
        )

    return feats


@register_extractor
class LibrosaFbank(FeatureExtractor):
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
