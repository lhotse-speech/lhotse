import io
import typing
from dataclasses import dataclass
from typing import ClassVar, List, Literal

import numpy as np

from lhotse.augmentation.transform import AudioTransform

Codec = Literal["opus", "mp3", "vorbis"]


@dataclass
class Compress(AudioTransform):
    """
    Modifies audio by running it through a lossy codec.

    :param codec: Used lossy audio codec. One of ``"opus"``, ``"mp3"``, or ``"vorbis"``.
    :param compression_level: The level of compression to apply. 0.0 is for the lowest amount of compression, 1.0 is for highest.
    :return: The modified audio samples.
    """

    supported_codecs: ClassVar[tuple[Codec]] = tuple(typing.get_args(Codec))
    codec: Codec
    compression_level: float

    def __call__(
        self,
        samples: np.ndarray,
        sampling_rate: int,
    ):
        import soundfile as sf

        # lhotse: (samples, channels)
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
            samples_compressed, sampling_rate_compressed = sf.read(f, always_2d=True)

        # when one writes Opus files with soundfile,
        # it adds extra information in the file header about the original sampling rate,
        # so that when we load this file with soundfile later,
        # it's resampled from 48k to original sampling rate
        # before returning the audio array (unlike most other tools)
        assert (
            sampling_rate_compressed == sampling_rate
        ), f"Sampling rate not preserved after compression: {sampling_rate_compressed} != {sampling_rate}"

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
        else:
            raise NotImplementedError(f"Unsupported augmentation codec {self.codec}")

    def reverse_timestamps(self, offset, duration, sampling_rate):
        return offset, duration
