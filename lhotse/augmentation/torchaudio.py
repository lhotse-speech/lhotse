import warnings
from dataclasses import dataclass
from decimal import ROUND_HALF_UP
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from lhotse.augmentation.resample import Resample as ResampleTensor
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
    global _precompiled_resamplers

    tpl = (source_sampling_rate, target_sampling_rate)
    if tpl not in _precompiled_resamplers:
        _precompiled_resamplers[tpl] = ResampleTensor(
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


class Codec:
    def __call__(self, samples: np.ndarray) -> np.ndarray:
        """
        Apply encoder then decoder.

        To be implemented in derived classes.
        """
        raise NotImplementedError


class MuLawCodec(Codec):
    def __init__(self):
        import torchaudio

        self.encoder = torchaudio.transforms.MuLawEncoding()
        self.decoder = torchaudio.transforms.MuLawDecoding()

    def __call__(self, samples):
        return self.decoder(self.encoder(samples))


from ctypes import CDLL, POINTER, c_int, c_short, c_uint8, c_void_p

LPC10_FRAME_SAMPLES = 180
LPC10_FRAME_BYTES = 7


def libspandsp_api():
    try:
        api = CDLL("libspandsp.so")
    except OSError as e:
        raise RuntimeError(
            "We cannot apply the narrowband transformation using the LPC10 codec as the SpanDSP library cannot be found. "
            "To install use `apt-get install libspandsp-dev` or visit <https://github.com/freeswitch/spandsp>."
        )

    api.lpc10_encode_init.restype = c_void_p
    api.lpc10_encode_init.argtypes = [c_void_p, c_int]

    api.lpc10_encode.restype = c_int
    api.lpc10_encode.argtypes = [c_void_p, POINTER(c_uint8), POINTER(c_short), c_int]

    api.lpc10_encode_free.argtypes = [c_void_p]

    api.lpc10_decode_init.restype = c_void_p
    api.lpc10_decode_init.argtypes = [c_void_p, c_int]

    api.lpc10_decode.restype = c_int
    api.lpc10_decode.argtypes = [c_void_p, POINTER(c_short), POINTER(c_uint8), c_int]

    api.lpc10_decode_free.argtypes = [c_void_p]

    return api


class LPC10Codec(Codec):
    def __init__(self):
        self.api = libspandsp_api()
        self.c_data = (c_uint8 * LPC10_FRAME_BYTES)()
        self.c_samples = (c_short * LPC10_FRAME_SAMPLES)()

    def __call__(self, samples):
        encoder = self.api.lpc10_encode_init(None, 0)
        decoder = self.api.lpc10_decode_init(None, 0)

        frames = samples[0].split(LPC10_FRAME_SAMPLES)

        idx = 0
        out = torch.zeros([1, len(frames) * LPC10_FRAME_SAMPLES])

        for frame in frames:

            samples_int = (frame * 32768).to(torch.int16)

            for i in range(0, samples_int.shape[0]):
                self.c_samples[i] = samples_int[i]

            for i in range(samples_int.shape[0], LPC10_FRAME_SAMPLES):
                self.c_samples[i] = 0

            assert (
                self.api.lpc10_encode(
                    encoder, self.c_data, self.c_samples, len(self.c_samples)
                )
                == LPC10_FRAME_BYTES
            )
            assert (
                self.api.lpc10_decode(
                    decoder, self.c_samples, self.c_data, LPC10_FRAME_BYTES
                )
                == LPC10_FRAME_SAMPLES
            )

            for i in range(0, LPC10_FRAME_SAMPLES):
                out[0][idx] = self.c_samples[i]
                idx = idx + 1

        self.api.lpc10_encode_free(encoder)
        self.api.lpc10_decode_free(decoder)

        return out / 32768


CODECS = {
    "lpc10": LPC10Codec,
    "mulaw": MuLawCodec,
}


@dataclass
class Narrowband(AudioTransform):
    """
    Narrowband effect.

    Resample input audio to 8000 Hz, apply codec (encode then immediately decode), then (optionally) resample back to the original sampling rate.
    """

    codec: str
    source_sampling_rate: int
    restore_orig_sr: bool

    def __post_init__(self):
        check_torchaudio_version()
        import torchaudio

        if self.codec in CODECS:
            self.codec_instance = CODECS[self.codec]()
        else:
            raise ValueError(f"unsupported codec: {self.codec}")

    def __call__(self, samples: np.ndarray, sampling_rate: int) -> np.ndarray:
        orig_size = samples.size

        samples = torch.from_numpy(samples)

        if self.source_sampling_rate != 8000:
            resampler_down = get_or_create_resampler(self.source_sampling_rate, 8000)
            samples = resampler_down(samples)

        samples = self.codec_instance(samples)

        if self.restore_orig_sr and self.source_sampling_rate != 8000:
            resampler_up = get_or_create_resampler(8000, self.source_sampling_rate)
            samples = resampler_up(samples)

        samples = samples.numpy()

        if self.restore_orig_sr and orig_size != samples.size:
            samples = np.resize(samples, (1, orig_size))

        return samples

    def reverse_timestamps(
        self,
        offset: Seconds,
        duration: Optional[Seconds],
        sampling_rate: Optional[int],
    ) -> Tuple[Seconds, Optional[Seconds]]:
        """
        This method just returnes the original offset and duration as the narrowband effect
        doesn't change any these audio properies.
        """

        return offset, duration


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
