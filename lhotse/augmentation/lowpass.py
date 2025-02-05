from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.signal
import soundfile as sf

from lhotse.augmentation.transform import AudioTransform


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
        N, _ = scipy.signal.kaiserord(ripple=100, width=0.05)
        taps = scipy.signal.firwin(
            numtaps=N,
            cutoff=self.frequency,
            width=0.05,
            fs=sampling_rate,
            pass_zero="lowpass",
        )
        return scipy.signal.lfilter(b=taps, a=1.0, x=samples)

    def reverse_timestamps(self, offset, duration, sampling_rate):
        return offset, duration
