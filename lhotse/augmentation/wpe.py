from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from lhotse.augmentation.transform import AudioTransform
from lhotse.utils import Seconds, is_module_available


@dataclass
class DereverbWPE(AudioTransform):
    """
    Dereverberation with Weighted Prediction Error (WPE).
    The implementation and default values are borrowed from `nara_wpe` package:
    https://github.com/fgnt/nara_wpe
    """

    n_fft: int = 512
    hop_length: int = 128
    taps: int = 10
    delay: int = 3
    iterations: int = 3
    statistics_mode: str = "full"

    def __call__(self, samples: np.ndarray, *args, **kwargs) -> np.ndarray:
        if isinstance(samples, np.ndarray):
            samples = torch.from_numpy(samples)
        augmented = dereverb_wpe_torch(samples, **asdict(self))
        return augmented.numpy()

    def reverse_timestamps(
        self, offset: Seconds, duration: Optional[Seconds], sampling_rate: int
    ) -> Tuple[Seconds, Optional[Seconds]]:
        return offset, duration


def dereverb_wpe_torch(
    audio: torch.Tensor,
    n_fft: int = 512,
    hop_length: int = 128,
    taps: int = 10,
    delay: int = 3,
    iterations: int = 3,
    statistics_mode: str = "full",
) -> torch.Tensor:
    if not is_module_available("nara_wpe"):
        raise ImportError(
            "Please install nara_wpe first using 'pip install git+https://github.com/fgnt/nara_wpe' "
            "(at the time of writing, only GitHub version has a PyTorch implementation)."
        )

    from nara_wpe.torch_wpe import wpe_v6

    assert audio.ndim == 2

    window = torch.blackman_window(n_fft)
    Y = torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        return_complex=True,
        window=window,
    )
    Y = Y.permute(1, 0, 2)
    Z = wpe_v6(
        Y,
        taps=taps,
        delay=delay,
        iterations=iterations,
        statistics_mode=statistics_mode,
    )
    z = torch.istft(
        Z.permute(1, 0, 2), n_fft=n_fft, hop_length=hop_length, window=window
    )
    return z
