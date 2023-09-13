import warnings
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from subprocess import PIPE, run
from typing import List, Optional, Union

import numpy as np

from lhotse.audio.backend import read_audio
from lhotse.audio.utils import (
    DurationMismatchError,
    get_audio_duration_mismatch_tolerance,
)
from lhotse.caching import AudioCache
from lhotse.utils import Pathlike, Seconds, SmartOpen, asdict_nonull, fastcopy


@dataclass
class AudioSource:
    """
    AudioSource represents audio data that can be retrieved from somewhere.
    """

    type: str
    """
    The type of audio source. Supported types are:
    - 'file' (supports most standard audio encodings, possibly multi-channel)
    - 'command' [unix pipe] (supports most standard audio encodings, possibly multi-channel)
    - 'url' (any URL type that is supported by "smart_open" library, e.g. http/https/s3/gcp/azure/etc.)
    - 'memory' (any format, read from a binary string attached to 'source' member of AudioSource)
    - 'shar' (indicates a placeholder that will be filled later when using Lhotse Shar data format)
    """

    channels: List[int]
    """
    A list of integer channel IDs available in this AudioSource.
    """

    source: Union[str, bytes]
    """
    The actual source to read from. The contents depend on the ``type`` field,
    but in general it can be a path, a URL, or the encoded binary data itself.
    """

    def load_audio(
        self,
        offset: Seconds = 0.0,
        duration: Optional[Seconds] = None,
        force_opus_sampling_rate: Optional[int] = None,
    ) -> np.ndarray:
        """
        Load the AudioSource (from files, commands, or URLs) with soundfile,
        accounting for many audio formats and multi-channel inputs.
        Returns numpy array with shapes: (n_samples,) for single-channel,
        (n_channels, n_samples) for multi-channel.

        Note: The elements in the returned array are in the range [-1.0, 1.0]
        and are of dtype `np.float32`.

        :param force_opus_sampling_rate: This parameter is only used when we detect an OPUS file.
            It will tell ffmpeg to resample OPUS to this sampling rate.
        """
        assert self.type in ("file", "command", "url", "memory", "shar")

        source = self.source

        if self.type == "command":
            if (offset != 0.0 or duration is not None) and not AudioCache.enabled():
                warnings.warn(
                    "You requested a subset of a recording that is read from disk via a bash command. "
                    "Expect large I/O overhead if you are going to read many chunks like these, "
                    "since every time we will read the whole file rather than its subset."
                    "You can use `lhotse.set_caching_enabled(True)` to mitigate the overhead."
                )

            # Let's assume 'self.source' is a pipe-command with unchangeable file,
            # never a microphone-stream or a live-stream.
            audio_bytes = AudioCache.try_cache(self.source)
            if not audio_bytes:
                audio_bytes = run(self.source, shell=True, stdout=PIPE).stdout
                AudioCache.add_to_cache(self.source, audio_bytes)

            samples, sampling_rate = read_audio(
                BytesIO(audio_bytes), offset=offset, duration=duration
            )

        elif self.type == "url":
            if offset != 0.0 or duration is not None and not AudioCache.enabled():
                warnings.warn(
                    "You requested a subset of a recording that is read from URL. "
                    "Expect large I/O overhead if you are going to read many chunks like these, "
                    "since every time we will download the whole file rather than its subset."
                    "You can use `lhotse.set_caching_enabled(True)` to mitigate the overhead."
                )

            # Let's assume 'self.source' is url to unchangeable file,
            # never a microphone-stream or a live-stream.
            audio_bytes = AudioCache.try_cache(self.source)
            if not audio_bytes:
                with SmartOpen.open(self.source, "rb") as f:
                    audio_bytes = f.read()
                AudioCache.add_to_cache(self.source, audio_bytes)

            samples, sampling_rate = read_audio(
                BytesIO(audio_bytes), offset=offset, duration=duration
            )

        elif self.type == "memory":
            assert isinstance(self.source, bytes), (
                "Corrupted manifest: specified AudioSource type is 'memory', "
                f"but 'self.source' attribute is not of type 'bytes' (found: '{type(self.source).__name__}')."
            )
            source = BytesIO(self.source)
            samples, sampling_rate = read_audio(
                source, offset=offset, duration=duration
            )
        elif self.type == "shar":
            raise RuntimeError(
                "Inconsistent state: found an AudioSource with Lhotse Shar placeholder "
                "that was not filled during deserialization."
            )

        else:  # self.type == 'file'
            samples, sampling_rate = read_audio(
                source,
                offset=offset,
                duration=duration,
                force_opus_sampling_rate=force_opus_sampling_rate,
            )

        # explicit sanity check for duration as soundfile does not complain here
        if duration is not None:
            num_samples = (
                samples.shape[0] if len(samples.shape) == 1 else samples.shape[1]
            )
            available_duration = num_samples / sampling_rate
            if (
                available_duration < duration - get_audio_duration_mismatch_tolerance()
            ):  # set the allowance as 1ms to avoid float error
                raise DurationMismatchError(
                    f"Requested more audio ({duration}s) than available ({available_duration}s)"
                )

        return samples.astype(np.float32)

    def with_path_prefix(self, path: Pathlike) -> "AudioSource":
        if self.type != "file":
            return self
        return fastcopy(self, source=str(Path(path) / self.source))

    def to_dict(self) -> dict:
        return asdict_nonull(self)

    @staticmethod
    def from_dict(data) -> "AudioSource":
        return AudioSource(**data)

    def __repr__(self):
        return (
            f"AudioSource(type='{self.type}', channels={self.channels}, "
            f"source='{self.source if isinstance(self.source, str) else '<binary-data>'}')"
        )
