from dataclasses import dataclass

import numpy as np

from lhotse.augmentation.transform import AudioTransform
from lhotse.utils import is_module_available


@dataclass
class Lowpass(AudioTransform):
    """
    Apply a low-pass filter to signal.

    :param frequency: The cutoff frequency of the low-pass filter.
    """

    frequency: float

    def __call__(
        self,
        samples: np.ndarray,
        sampling_rate: int,
    ):
        if not is_module_available("scipy"):
            raise ImportError(
                "In order to use Lowpass transforms, run 'pip install scipy'"
            )

        import scipy.signal

        width = 0.02
        N, _ = scipy.signal.kaiserord(ripple=200, width=width)
        taps = scipy.signal.firwin(
            numtaps=N,
            cutoff=self.frequency,
            width=width,
            fs=sampling_rate,
            pass_zero="lowpass",
        )
        return scipy.signal.lfilter(b=taps, a=1.0, x=samples)

    def reverse_timestamps(self, offset, duration, sampling_rate):
        return offset, duration
