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
        if self.codec == "gsm" and sampling_rate != 8000:
            raise ValueError(
                f"GSM codec only supports 8000Hz sampling rate, got {sampling_rate}"
            )
        if self.codec == "opus" and sampling_rate not in OPUS_SUPPORTED_SAMPLING_RATES:
            raise ValueError(
                f"Opus codec only supports {','.join(map(str, OPUS_SUPPORTED_SAMPLING_RATES))}, got {sampling_rate}"
            )
        if self.codec == "mp3" and sampling_rate not in MP3_SUPPORTED_SAMPLING_RATES:
            raise ValueError(
                f"MP3 codec only supports {','.join(map(str, MP3_SUPPORTED_SAMPLING_RATES))}, got {sampling_rate}"
            )

        import soundfile as sf

        # lhotse: (samples, channels)
        _, channels = samples.shape

        # soundfile: (channels, samples)
        samples = samples.transpose(1, 0)

        with io.BytesIO() as f:
            sf.write(
                f,
                samples,
                sampling_rate,
                closefd=False,
                compression_level=self.compression_level,
                **self.prepare_sf_arguments(),
            )
            f.seek(0)
            if self.codec == "gsm":
                samples_compressed, sampling_rate_compressed = sf.read(
                    f,
                    always_2d=True,
                    samplerate=sampling_rate,
                    channels=channels,
                    format="RAW",
                    subtype="GSM610",
                )
            else:
                samples_compressed, sampling_rate_compressed = sf.read(
                    f, always_2d=True
                )

        # when one writes Opus files with soundfile,
        # it adds extra information in the file header about the original sampling rate,
        # so that when we load this file with soundfile later,
        # it's resampled from 48k to original sampling rate
        # before returning the audio array (unlike most other tools)
        assert (
            sampling_rate_compressed == sampling_rate
        ), f"Sampling rate not preserved after compression: compressed {sampling_rate_compressed} != original {sampling_rate}"

        samples_compressed = samples_compressed.transpose(1, 0)

        return samples_compressed

    def prepare_sf_arguments(self) -> dict:
        import soundfile as sf

        if self.codec == "mp3":
            return {"format": "MP3", "subtype": sf.default_subtype("MP3")}
        elif self.codec == "opus":
            return {"format": "OGG", "subtype": "OPUS"}
        elif self.codec == "vorbis":
            return {"format": "OGG", "subtype": "VORBIS"}
        elif self.codec == "gsm":
            return {"format": "RAW", "subtype": "GSM610"}
        else:
            raise NotImplementedError(f"Unsupported augmentation codec {self.codec}")

    def reverse_timestamps(self, offset, duration, sampling_rate):
        return offset, duration
