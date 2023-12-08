import codecs
import json
from functools import partial
from io import BytesIO
from typing import List, Optional

import numpy as np
import torch
from typing_extensions import Literal

from lhotse import Recording
from lhotse.shar.utils import to_shar_placeholder
from lhotse.shar.writers.tar import TarWriter


class AudioTarWriter:
    """
    AudioTarWriter writes audio examples in numpy arrays or PyTorch tensors into a tar archive
    that is automatically sharded.

    It is different from :class:`~lhotse.shar.writers.array.ArrayTarWriter` in that it supports
    audio-specific compression mechanisms, such as ``flac`` or ``mp3``.

    Example::

        >>> with AudioTarWriter("some_dir/audio.%06d.tar", shard_size=100, format="mp3") as w:
        ...     w.write("audio1", audio1_array)
        ...     w.write("audio2", audio2_array)  # etc.

    It would create files such as ``some_dir/audio.000000.tar``, ``some_dir/audio.000001.tar``, etc.

    It's also possible to use ``AudioTarWriter`` with automatic sharding disabled::

        >>> with AudioTarWriter("some_dir/audio.tar", shard_size=None, format="flac") as w:
        ...     w.write("audio1", audio1_array)
        ...     w.write("audio2", audio2_array)  # etc.

    See also: :class:`~lhotse.shar.writers.tar.TarWriter`, :class:`~lhotse.shar.writers.array.ArrayTarWriter`
    """

    def __init__(
        self,
        pattern: str,
        shard_size: Optional[int] = 1000,
        format: Literal["wav", "flac", "mp3"] = "flac",
    ):
        import torchaudio

        self.format = format
        self.tar_writer = TarWriter(pattern, shard_size)
        self.save_fn = torchaudio.save
        if self.format == "flac":
            self.save_fn = partial(
                torchaudio.backend.soundfile_backend.save, bits_per_sample=16
            )

    def __enter__(self):
        self.tar_writer.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.tar_writer.close()

    @property
    def output_paths(self) -> List[str]:
        return self.tar_writer.output_paths

    def write_placeholder(self, key: str) -> None:
        self.tar_writer.write(f"{key}.nodata", BytesIO())
        self.tar_writer.write(f"{key}.nometa", BytesIO(), count=False)

    def write(
        self,
        key: str,
        value: np.ndarray,
        sampling_rate: int,
        manifest: Recording,
    ) -> None:
        # Write binary data
        stream = BytesIO()
        self.save_fn(
            stream,
            torch.from_numpy(value),
            sampling_rate,
            format=self.format,
        )
        self.tar_writer.write(f"{key}.{self.format}", stream)

        # Write text manifest afterwards
        manifest = to_shar_placeholder(manifest)
        json_stream = BytesIO()
        print(
            json.dumps(manifest.to_dict()),
            file=codecs.getwriter("utf-8")(json_stream),
        )
        json_stream.seek(0)
        self.tar_writer.write(f"{key}.json", json_stream, count=False)
