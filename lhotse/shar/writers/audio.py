import codecs
import json
from io import BytesIO
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import torch

from lhotse import Recording
from lhotse.audio.backend import (
    LibsndfileBackend,
    get_current_audio_backend,
    save_audio,
)
from lhotse.augmentation import get_or_create_resampler
from lhotse.shar.utils import to_shar_placeholder
from lhotse.shar.writers.tar import TarWriter
from lhotse.utils import is_torchaudio_available


class AudioTarWriter:
    """
    AudioTarWriter writes audio examples in numpy arrays or PyTorch tensors into a tar archive
    that is automatically sharded.

    It is different from :class:`~lhotse.shar.writers.array.ArrayTarWriter` in that it supports
    audio-specific compression mechanisms, such as ``flac``, ``opus``, ``mp3``, or ``wav``.

    Example::

        >>> with AudioTarWriter("some_dir/audio.%06d.tar", shard_size=100, format="mp3") as w:
        ...     w.write("audio1", audio1_array)
        ...     w.write("audio2", audio2_array)  # etc.

    It would create files such as ``some_dir/audio.000000.tar``, ``some_dir/audio.000001.tar``, etc.
    The starting shard offset can be set using ``shard_offset`` parameter. The writer starts from 0 by default.

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
        format: Literal["wav", "flac", "mp3", "opus"] = "flac",
        shard_offset: int = 0,
    ):
        self.format = format
        self.tar_writer = TarWriter(pattern, shard_size, shard_offset=shard_offset)

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

    def resolve_format(self, original_format: str):
        if self.format == "original":
            # save using the original format of the input audio
            return original_format
        else:
            # save using the format specified at initialization
            return self.format

    def write_placeholder(self, key: str) -> None:
        self.tar_writer.write(f"{key}.nodata", BytesIO())
        self.tar_writer.write(f"{key}.nometa", BytesIO(), count=False)

    def write(
        self,
        key: str,
        value: np.ndarray,
        sampling_rate: int,
        manifest: Recording,
        original_format: Optional[str] = None,
    ) -> None:
        save_format = self.resolve_format(original_format)

        value, manifest, sampling_rate = self._maybe_resample(
            value, manifest, sampling_rate, format=save_format
        )

        # Write binary data
        stream = BytesIO()
        save_audio(
            dest=stream, src=value, sampling_rate=sampling_rate, format=save_format
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

    def _maybe_resample(
        self,
        audio: Union[torch.Tensor, np.ndarray],
        manifest: Recording,
        sampling_rate: int,
        format: str,
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Recording, int]:
        # Resampling is required for some versions of OPUS encoders.
        # First resample the manifest which only adjusts the metadata;
        # then resample the audio array to 48kHz.
        OPUS_DEFAULT_SAMPLING_RATE = 48000
        if (
            format == "opus"
            and is_torchaudio_available()
            and not isinstance(get_current_audio_backend(), LibsndfileBackend)
            and sampling_rate != OPUS_DEFAULT_SAMPLING_RATE
        ):
            manifest = manifest.resample(OPUS_DEFAULT_SAMPLING_RATE)
            audio = get_or_create_resampler(sampling_rate, OPUS_DEFAULT_SAMPLING_RATE)(
                torch.as_tensor(audio)
            )
            return audio, manifest, OPUS_DEFAULT_SAMPLING_RATE
        return audio, manifest, sampling_rate
