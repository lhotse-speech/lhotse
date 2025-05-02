import typing
from dataclasses import dataclass
from typing import ClassVar, Literal, Tuple

import numpy as np

from lhotse.augmentation.transform import AudioTransform
from lhotse.utils import is_module_available

Filter = Literal["butter", "cheby1", "cheby2", "ellip", "bessel"]


@dataclass
class Lowpass(AudioTransform):
    """
    Apply a low-pass filter to signal.

    :param frequency: The cutoff frequency of the low-pass filter.
    :param filter_type: Type of filter to use. One of ['butter', 'cheby1', 'cheby2', 'ellip', 'bessel']
    :param order: The order of the filter (default: 4)
    :param ripple_db: Passband ripple in dB for Chebyshev I and Elliptical filters (default: 0.1)
    :param stopband_attenuation: Stopband attenuation in dB for Chebychev II and Elliptical filters (default: 40)
    """

    supported_filters: ClassVar[Tuple[Filter]] = tuple(typing.get_args(Filter))
    frequency: float
    filter_type: str = "butter"
    order: int = 4
    ripple_db: float = 0.1
    stopband_attenuation_db: float = 40

    def __post_init__(self):
        if self.filter_type not in self.supported_filters:
            raise ValueError(
                f"Filter type '{self.filter_type}' is not supported. "
                f"Supported types are: {self.supported_filters}"
            )
        if self.frequency <= 0:
            raise ValueError("Cutoff frequency must be positive")
        if self.order <= 0:
            raise ValueError("Filter order must be positive")
        if self.ripple_db <= 0:
            raise ValueError("Passband ripple must be positive")
        if self.stopband_attenuation_db <= 0:
            raise ValueError("Stopband attenuation must be positive")

    def __call__(
        self,
        samples: np.ndarray,
        sampling_rate: int,
    ) -> np.ndarray:
        if self.frequency >= sampling_rate / 2:
            raise ValueError(
                f"Cutoff frequency ({self.frequency} Hz) must be less than half the sampling rate ({sampling_rate/2} Hz)"
            )

        if not is_module_available("scipy"):
            raise ImportError(
                "In order to use Lowpass transforms, run 'pip install scipy'"
            )
        import scipy.signal

        common_kwargs = {
            "N": self.order,
            "Wn": self.frequency,
            "btype": "low",
            "output": "sos",
            "fs": sampling_rate,
        }

        if self.filter_type == "butter":
            sos = scipy.signal.butter(**common_kwargs)
        elif self.filter_type == "cheby1":
            sos = scipy.signal.cheby1(rp=self.ripple_db, **common_kwargs)
        elif self.filter_type == "cheby2":
            sos = scipy.signal.cheby2(rs=self.stopband_attenuation_db, **common_kwargs)
        elif self.filter_type == "ellip":
            sos = scipy.signal.ellip(
                rp=self.ripple_db, rs=self.stopband_attenuation_db, **common_kwargs
            )
        elif self.filter_type == "bessel":
            sos = scipy.signal.bessel(**common_kwargs)
        else:
            raise Exception(f"The lowpass filter {self.filter_type} is not supported!")

        return scipy.signal.sosfiltfilt(
            sos, samples
        ).copy()  # copy because torch.from_numpy complains

    def reverse_timestamps(self, offset, duration, sampling_rate):
        return offset, duration
