import io
from dataclasses import dataclass
from typing import Literal

import numpy as np
import soundfile as sf

from lhotse.augmentation.transform import AudioTransform


@dataclass
class Compress(AudioTransform):
    """
    Modifies audio by running it through a lossy codec.

    :param codec: Used lossy audio codec. One of ``"opus"``, ``"mp3"``, or ``"vorbis"``.
    :param compression_level: The level of compression to apply. 0.0 is for the lowest amount of compression, 1.0 is for highest.
    :return: The modified audio samples.
    """

    codec: Literal["opus", "mp3", "vorbis"]
    compression_level: float

    def __call__(
        self,
        samples: np.ndarray,
        sampling_rate: int,
    ):
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
            samples_compressed, rate_compressed = sf.read(
                f, always_2d=True
            )  # TODO: handle possible sample rate change with the opus codec?
        samples_compressed = samples_compressed.transpose(1, 0)

        return samples_compressed

    def prepare_sf_arguments(self) -> dict:
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
