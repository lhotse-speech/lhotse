from dataclasses import asdict, dataclass
from typing import Optional, Tuple, Union

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

    def __call__(
        self, samples: Union[np.ndarray, torch.Tensor], *args, **kwargs
    ) -> np.ndarray:
        if torch.is_tensor(samples):
            samples = samples.cpu().numpy()
        augmented = dereverb_wpe_numpy(samples, **asdict(self))
        return augmented

    def reverse_timestamps(
        self, offset: Seconds, duration: Optional[Seconds], sampling_rate: int
    ) -> Tuple[Seconds, Optional[Seconds]]:
        return offset, duration


def dereverb_wpe_numpy(
    audio: np.ndarray,
    n_fft: int = 512,
    hop_length: int = 128,
    taps: int = 10,
    delay: int = 3,
    iterations: int = 3,
    statistics_mode: str = "full",
) -> np.ndarray:
    """
    Applies WPE-based dereverberation using nara_wpe's wpe_v8 function with numpy backend.
    The parameter defaults follow the ones in nara_wpe.
    """
    if not is_module_available("nara_wpe"):
        raise ImportError(
            "Please install nara_wpe first using 'pip install git+https://github.com/fgnt/nara_wpe'"
        )

    from nara_wpe.wpe import wpe_v8

    assert audio.ndim == 2, f"Expected 2D audio shape, got: {audio.shape}"

    window = torch.blackman_window(n_fft)
    Y = torch.stft(
        torch.from_numpy(audio),
        n_fft=n_fft,
        hop_length=hop_length,
        return_complex=True,
        window=window,
    )
    Y = Y.permute(1, 0, 2).numpy()
    Z = wpe_v8(
        Y,
        taps=taps,
        delay=delay,
        iterations=iterations,
        statistics_mode=statistics_mode,
    )
    z = torch.istft(
        torch.from_numpy(Z).permute(1, 0, 2),
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
    ).numpy()
    return z


def dereverb_wpe_torch(
    audio: torch.Tensor,
    n_fft: int = 512,
    hop_length: int = 128,
    taps: int = 10,
    delay: int = 3,
    iterations: int = 3,
    statistics_mode: str = "full",
) -> torch.Tensor:
    """
    Applies WPE-based dereverberation using nara_wpe's wpe_v6 function with PyTorch backend.
    The parameter defaults follow the ones in nara_wpe.

    .. caution:: The PyTorch backend is known to sometimes be less stable than the numpy backend.
    """
    if not is_module_available("nara_wpe"):
        raise ImportError(
            "Please install nara_wpe first using 'pip install git+https://github.com/fgnt/nara_wpe'"
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
