from functools import partial
from io import BytesIO
from typing import Literal

import numpy as np
import torch
import torchaudio

from lhotse.shar.writers.tar import TarWriter


class AudioTarWriter:
    def __init__(
        self,
        pattern: str,
        shard_size: int,
        format: Literal["wav", "flac", "mp3"] = "flac",
    ):
        self.format = format
        self.tar_writer = TarWriter(pattern, shard_size)
        self.save_fn = torchaudio.save
        if self.format == "flac":
            self.save_fn = partial(
                torchaudio.backend.soundfile_backend.save, bits_per_sample=16
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.tar_writer.close()

    def write(self, key: str, value: np.ndarray, sampling_rate: int) -> str:
        stream = BytesIO()
        self.save_fn(
            stream,
            torch.from_numpy(value),
            sampling_rate,
            format=self.format,
        )
        self.tar_writer.write(f"{key}.{self.format}", stream)
