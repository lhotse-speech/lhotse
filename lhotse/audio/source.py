import warnings
from dataclasses import dataclass
from io import BytesIO, FileIO
from pathlib import Path
from subprocess import PIPE, run
from typing import List, Optional, Union

import numpy as np
import torch

from lhotse.audio.backend import read_audio
from lhotse.audio.utils import (
    DurationMismatchError,
    get_audio_duration_mismatch_tolerance,
)
from lhotse.caching import AudioCache
from lhotse.utils import (
    Pathlike,
    Seconds,
    SmartOpen,
    asdict_nonull,
    compute_num_samples,
    fastcopy,
)

PathOrFilelike = Union[str, BytesIO, FileIO]


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
        source = self._prepare_for_reading(offset=offset, duration=duration)

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

    def load_video(
        self, offset: Seconds = 0.0, duration: Optional[Seconds] = None
    ) -> torch.Tensor:
        import torchaudio

        # Open the video file for reading.
        stream = torchaudio.io.StreamReader(self.source)

        # Collect the information about available video and audio streams.
        num_streams = stream.num_src_streams
        audio_streams = {}
        video_streams = {}
        for stream_idx in range(num_streams):
            info = stream.get_src_stream_info(stream_idx)
            if info.media_type == "video":
                video_streams[stream_idx] = info
            elif info.media_type == "audio":
                audio_streams[stream_idx] = info
            else:
                raise RuntimeError(f"Unexpected media_type: {info}")
        assert (
            len(video_streams) != 0
        ), "The file does not seem to have any video streams."
        assert (
            len(video_streams) < 2
        ), f"Lhotse currently does not support more than one video stream in a file (found {len(video_streams)})."

        # Add an ffmpeg output video stream to perform reading in chunks.
        ((video_stream_idx, video_stream),) = list(video_streams.items())
        stream.add_basic_video_stream(
            # TODO: check if this creates an issue with 29.97... etc fps, after long time it might desync the audio?
            round(video_stream.frame_rate),
            stream_index=video_stream_idx,
        )

        # # Pre-allocate the memory for output video to reduce the peak CPU mem usage
        # output_video = torch.empty(
        #     (video_stream.num_frames, 3, video_stream.height, video_stream.width),
        #     dtype=torch.uint8,
        # )

        stream.seek(offset)
        chunks = []
        # frame_offset = 0
        decoded_duration = 0.0
        for (video_chunk,) in stream.stream():
            if duration is not None and decoded_duration >= duration:
                break

            chunk_size = video_chunk.size(0)
            chunk_duration = chunk_size / video_stream.frame_rate

            if duration is not None and decoded_duration + chunk_duration > duration:
                keep_frames = compute_num_samples(
                    1.0 - chunk_duration, video_stream.frame_rate
                )
                video_chunk = video_chunk[:keep_frames]

            # output_video[frame_offset : frame_offset + chunk_size] = video_chunk
            chunks.append(video_chunk)
            decoded_duration += chunk_duration
            # frame_offset += chunk_size

        if not chunks:
            return torch.zeros(
                0, 3, video_stream.height, video_stream.width, dtype=torch.uint8
            )

        output_video = torch.cat(chunks, dim=0)

        return output_video

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

    def _prepare_for_reading(
        self, offset: Seconds, duration: Optional[Seconds]
    ) -> PathOrFilelike:
        """
        Validates `self.type` and prepares the actual source for audio reading.
        Returns either a path or a file-like object opened in binary mode,
        that can be handled by :func:`lhotse.audio.backend.read_audio`.
        """
        assert self.type in (
            "file",
            "command",
            "url",
            "memory",
            "shar",
        ), f"Unexpected AudioSource type: '{self.type}'"

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
            source = BytesIO(audio_bytes)

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
            source = BytesIO(audio_bytes)

        elif self.type == "memory":

            assert isinstance(self.source, bytes), (
                "Corrupted manifest: specified AudioSource type is 'memory', "
                f"but 'self.source' attribute is not of type 'bytes' (found: '{type(self.source).__name__}')."
            )
            source = BytesIO(self.source)

        elif self.type == "shar":

            raise RuntimeError(
                "Inconsistent state: found an AudioSource with Lhotse Shar placeholder "
                "that was not filled during deserialization."
            )

        return source
