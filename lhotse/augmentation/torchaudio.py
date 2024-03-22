import warnings
from dataclasses import dataclass
from decimal import ROUND_HALF_UP
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from lhotse.augmentation.transform import AudioTransform
from lhotse.utils import (
    Seconds,
    compute_num_samples,
    during_docs_build,
    is_module_available,
    is_torchaudio_available,
    perturb_num_samples,
)


@dataclass
class Speed(AudioTransform):
    """
    Speed perturbation effect, the same one as invoked with `sox speed` in the command line.

    It resamples the signal back to the input sampling rate, so the number of output samples will
    be smaller or greater, depending on the speed factor.
    """

    factor: float

    def __call__(self, samples: np.ndarray, sampling_rate: int) -> np.ndarray:
        check_for_torchaudio()
        resampler = get_or_create_resampler(
            round(sampling_rate * self.factor), sampling_rate
        )
        augmented = resampler(torch.from_numpy(samples))
        return augmented.numpy()

    def reverse_timestamps(
        self, offset: Seconds, duration: Optional[Seconds], sampling_rate: int
    ) -> Tuple[Seconds, Optional[Seconds]]:
        """
        This method helps estimate the original offset and duration for a recording
        before speed perturbation was applied.
        We need this estimate to know how much audio to actually load from disk during the
        call to ``load_audio()``.
        """
        start_sample = compute_num_samples(offset, sampling_rate)
        num_samples = (
            compute_num_samples(duration, sampling_rate)
            if duration is not None
            else None
        )
        start_sample = perturb_num_samples(start_sample, 1 / self.factor)
        num_samples = (
            perturb_num_samples(num_samples, 1 / self.factor)
            if num_samples is not None
            else None
        )
        return (
            start_sample / sampling_rate,
            num_samples / sampling_rate if num_samples is not None else None,
        )


_precompiled_resamplers: Dict[Tuple[int, int], torch.nn.Module] = {}


def get_or_create_resampler(
    source_sampling_rate: int, target_sampling_rate: int
) -> torch.nn.Module:
    check_for_torchaudio()
    global _precompiled_resamplers

    tpl = (source_sampling_rate, target_sampling_rate)
    if tpl not in _precompiled_resamplers:
        check_torchaudio_version()
        import torchaudio

        _precompiled_resamplers[tpl] = torchaudio.transforms.Resample(
            source_sampling_rate, target_sampling_rate
        )
    return _precompiled_resamplers[tpl]


@dataclass
class Resample(AudioTransform):
    """
    Resampling effect, the same one as invoked with `sox rate` in the command line.
    """

    source_sampling_rate: int
    target_sampling_rate: int

    def __post_init__(self):
        self.source_sampling_rate = int(self.source_sampling_rate)
        self.target_sampling_rate = int(self.target_sampling_rate)
        if not is_torchaudio_available():
            assert is_module_available(
                "scipy"
            ), "In order to use resampling, either torchaudio or scipy needs to be installed."
        else:
            self.resampler = get_or_create_resampler(
                self.source_sampling_rate, self.target_sampling_rate
            )

    def __call__(self, samples: np.ndarray, *args, **kwargs) -> np.ndarray:
        if self.source_sampling_rate == self.target_sampling_rate:
            return samples

        if is_torchaudio_available():
            if isinstance(samples, np.ndarray):
                samples = torch.from_numpy(samples)
            augmented = self.resampler(samples)
            return augmented.numpy()
        else:
            import scipy

            gcd = np.gcd(self.source_sampling_rate, self.target_sampling_rate)
            augmented = scipy.signal.resample_poly(
                samples,
                up=self.target_sampling_rate // gcd,
                down=self.source_sampling_rate // gcd,
                axis=-1,
            )
            return augmented

    def reverse_timestamps(
        self, offset: Seconds, duration: Optional[Seconds], sampling_rate: int
    ) -> Tuple[Seconds, Optional[Seconds]]:
        """
        This method helps estimate the original offset and duration for a recording
        before resampling was applied.
        We need this estimate to know how much audio to actually load from disk during the
        call to ``load_audio()``.

        In case of resampling, the timestamps might change slightly when using non-trivial
        pairs of sampling rates, e.g. 16kHz -> 22.05kHz, because the number of samples in
        the resampled audio might actually correspond to incrementally larger/smaller duration.
        E.g. 16kHz, 235636 samples correspond to 14.72725s duration; after resampling to 22.05kHz,
        it is 324736 samples which correspond to 14.727256235827664s duration.
        """
        if self.source_sampling_rate == self.target_sampling_rate:
            return offset, duration

        old_num_samples = compute_num_samples(
            offset, self.source_sampling_rate, rounding=ROUND_HALF_UP
        )
        old_offset = old_num_samples / self.source_sampling_rate
        if duration is not None:
            old_num_samples = compute_num_samples(
                duration, self.source_sampling_rate, rounding=ROUND_HALF_UP
            )
            old_duration = old_num_samples / self.source_sampling_rate
        else:
            old_duration = None
        return old_offset, old_duration


@dataclass
class Tempo(AudioTransform):
    """Tempo perturbation effect, the same one as invoked with `sox tempo` in the command line.

    Compared to speed perturbation, tempo preserves pitch.
    It resamples the signal back to the input sampling rate, so the number of output samples will
    be smaller or greater, depending on the tempo factor.
    """

    factor: float

    def __call__(self, samples: np.ndarray, sampling_rate: int) -> np.ndarray:
        check_for_torchaudio()
        check_torchaudio_version()
        import torchaudio

        sampling_rate = int(sampling_rate)  # paranoia mode
        if isinstance(samples, np.ndarray):
            samples = torch.from_numpy(samples)

        augmented, new_sampling_rate = torchaudio.sox_effects.apply_effects_tensor(
            samples, sampling_rate, [["tempo", str(self.factor)]]
        )
        return augmented.numpy()

    def reverse_timestamps(
        self,
        offset: Seconds,
        duration: Optional[Seconds],
        sampling_rate: int,
    ) -> Tuple[Seconds, Optional[Seconds]]:
        """
        This method helps estimate the original offset and duration for a recording
        before tempo perturbation was applied.
        We need this estimate to know how much audio to actually load from disk during the
        call to ``load_audio()``.
        """
        start_sample = compute_num_samples(offset, sampling_rate)
        num_samples = (
            compute_num_samples(duration, sampling_rate)
            if duration is not None
            else None
        )
        start_sample = perturb_num_samples(start_sample, 1 / self.factor)
        num_samples = (
            perturb_num_samples(num_samples, 1 / self.factor)
            if num_samples is not None
            else None
        )
        return (
            start_sample / sampling_rate,
            num_samples / sampling_rate if num_samples is not None else None,
        )


@dataclass
class Volume(AudioTransform):
    """
    Volume perturbation effect, the same one as invoked with `sox vol` in the command line.
    It applies given gain (factor) to the input, without any postprocessing (such as a limiter).
    """

    factor: float

    def __call__(self, samples: np.ndarray, sampling_rate: int) -> np.ndarray:
        # We only support the simplest case in SoX which is multiplication by a gain value.
        # https://github.com/chirlu/sox/blob/42b3557e13e0fe01a83465b672d89faddbe65f49/src/vol.c#L149
        return samples * self.factor

    def reverse_timestamps(
        self,
        offset: Seconds,
        duration: Optional[Seconds],
        sampling_rate: Optional[int],  # Not used, made for compatibility purposes
    ) -> Tuple[Seconds, Optional[Seconds]]:
        """
        This method just returnes the original offset and duration as volume perturbation
        doesn't change any these audio properies.
        """
        return offset, duration


def check_torchaudio_version():
    import torchaudio
    from packaging.version import parse as _version

    if not during_docs_build() and _version(torchaudio.__version__) < _version("0.7"):
        warnings.warn(
            "Torchaudio SoX effects chains are only introduced in version 0.7 - "
            "please upgrade your PyTorch to 1.7.1 and torchaudio to 0.7.2 (or higher) "
            "to use them."
        )


def check_for_torchaudio():
    if not is_torchaudio_available():
        raise RuntimeError(
            "This transform is not supported in torchaudio-free Lhotse installation. "
            "Please install torchaudio and try again."
        )
