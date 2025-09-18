from dataclasses import dataclass
from typing import Optional

import numpy as np

from lhotse.augmentation.transform import AudioTransform


@dataclass
class Clipping(AudioTransform):
    """
    Apply clipping to audio signal.

    This augmentation simulates the effect of clipping that occurs when audio levels exceed
    the dynamic range of recording equipment or when audio is amplified beyond its limits.

    Clips input signal to [-1, 1] range.

    :param gain_db: The amount of gain in decibels to apply before clipping (and to revert back to original level after).
    :param hard: If True, apply hard clipping (sharp cutoff); otherwise, apply soft clipping (saturation).
    :param normalize: If True, normalize the input signal to 0 dBFS before applying clipping.
    """

    hard: bool = False
    gain_db: float = 0.0
    normalize: bool = True

    def __call__(
        self,
        samples: np.ndarray,
        sampling_rate: int,
    ) -> np.ndarray:
        max_peak_amplitude = np.max(np.abs(samples))

        # Treat signals with extremely low amplitude (below -96 dBFS peak) as silence
        if max_peak_amplitude == 0 or 20 * np.log10(max_peak_amplitude) < -96:
            return samples.copy()

        # Normalize to 0 dBFS
        if self.normalize:
            samples = samples / max_peak_amplitude

        # Apply gain
        if abs(self.gain_db) >= 0.1:
            gain_linear = 10 ** (self.gain_db / 20.0)
            samples = samples * gain_linear

        if self.hard:
            saturated_samples = np.clip(samples, -1.0, 1.0)
        else:
            saturated_samples = np.tanh(samples)

        # Revert the additional gain applied before clipping
        if abs(self.gain_db) >= 0.1:
            saturated_samples = saturated_samples / gain_linear

        # Revert 0 dBFS normalization
        if self.normalize:
            saturated_samples = saturated_samples * max_peak_amplitude

        return saturated_samples.copy()

    def reverse_timestamps(self, offset, duration, sampling_rate):
        """
        Clipping doesn't change the timing of the audio, so timestamps remain unchanged.
        """
        return offset, duration
