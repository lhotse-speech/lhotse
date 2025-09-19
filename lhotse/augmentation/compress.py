import io
import logging
import typing
from dataclasses import dataclass
from typing import ClassVar, Literal, Optional, Tuple

import numpy as np

from lhotse.augmentation.transform import AudioTransform

Codec = Literal["opus", "mp3", "vorbis", "gsm"]

OPUS_SUPPORTED_SAMPLING_RATES = [8000, 12000, 16000, 24000, 48000]
MP3_SUPPORTED_SAMPLING_RATES = [
    8000,
    11025,
    12000,
    16000,
    22050,
    24000,
    32000,
    44100,
    48000,
]


@dataclass
class Compress(AudioTransform):
    """
    Modifies audio by running it through a lossy codec.

    :param codec: Used lossy audio codec. One of ``"opus"``, ``"mp3"``, ``"vorbis"``, or ``"gsm"``.
    :param compression_level: The level of compression to apply. 0.0 is for the lowest amount of compression, 1.0 is for highest. Ignored for ``"gsm"``.
    :return: The modified audio samples.
    """

    supported_codecs: ClassVar[Tuple[Codec]] = tuple(typing.get_args(Codec))
    codec: Codec
    compression_level: Optional[float] = None

    def __post_init__(self):
        if self.codec not in self.supported_codecs:
            raise ValueError(f"Unsupported augmentation codec {self.codec}")
        if not 0 <= self.compression_level <= 1:
            raise ValueError("Compression level must be between 0 and 1")

    def __call__(
        self,
        samples: np.ndarray,
        sampling_rate: int,
    ):
        # argument `sample_rate` is the original sampling rate of the `Recording` before any transforms are applied
        # even if we Resample prior to Compress, it is useless to check for it here
        # we just assume 8 kHz for GSM and original sampling rate for other codecs

        import soundfile as sf

        # lhotse: (channels, samples)
        # soundfile: (samples, channels)

        channels, _ = samples.shape
        samples = samples.transpose(1, 0)

        with io.BytesIO() as buffer:
            sf.write(
                buffer,
                samples,
                samplerate=sampling_rate if self.codec != "gsm" else 8000,
                closefd=False,
                **self.prepare_sf_arguments(),
            )
            data = buffer.getvalue()
        with io.BytesIO(data) as f:
            if self.codec == "gsm":
                samples_compressed, sampling_rate_compressed = sf.read(
                    f,
                    always_2d=True,
                    samplerate=sampling_rate if self.codec != "gsm" else 8000,
                    channels=channels,
                    format="RAW",
                    subtype="GSM610",
                    dtype=np.float32,
                )
            else:
                samples_compressed, sampling_rate_compressed = sf.read(
                    f, always_2d=True, dtype=np.float32
                )

        # when one writes Opus files with soundfile,
        # it adds extra information in the file header about the original sampling rate,
        # so that when we load this file with soundfile later,
        # it's resampled to original sampling rate
        # before returning the audio array (unlike most other tools)

        samples_compressed = samples_compressed.transpose(1, 0)

        return samples_compressed

    def prepare_sf_arguments(self) -> dict:
        import soundfile as sf

        if self.codec == "mp3":
            return {
                "compression_level": self.compression_level,
                "format": "MP3",
                "subtype": sf.default_subtype("MP3"),
            }
        elif self.codec == "opus":
            return {
                "compression_level": self.compression_level,
                "format": "OGG",
                "subtype": "OPUS",
            }
        elif self.codec == "vorbis":
            return {
                "compression_level": self.compression_level,
                "format": "OGG",
                "subtype": "VORBIS",
            }
        elif self.codec == "gsm":
            return {"format": "RAW", "subtype": "GSM610"}
        else:
            raise NotImplementedError(f"Unsupported augmentation codec {self.codec}")

    def reverse_timestamps(self, offset, duration, sampling_rate):
        return offset, duration
