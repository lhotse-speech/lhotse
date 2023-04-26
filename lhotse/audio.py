import functools
import logging
import os
import random
import re
import sys
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import ROUND_HALF_UP
from functools import lru_cache, partial
from io import BytesIO, IOBase
from itertools import islice
from math import ceil, sqrt
from pathlib import Path
from subprocess import PIPE, CalledProcessError, run
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import torch
from tqdm.auto import tqdm

from lhotse.augmentation import (
    AudioTransform,
    DereverbWPE,
    LoudnessNormalization,
    Resample,
    ReverbWithImpulseResponse,
    Speed,
    Tempo,
    Volume,
)
from lhotse.caching import dynamic_lru_cache
from lhotse.lazy import AlgorithmMixin
from lhotse.serialization import Serializable
from lhotse.utils import (
    Decibels,
    NonPositiveEnergyError,
    Pathlike,
    Seconds,
    SetContainingAnything,
    SmartOpen,
    asdict_nonull,
    compute_num_samples,
    exactly_one_not_null,
    fastcopy,
    ifnone,
    index_by_id_and_check,
    perturb_num_samples,
    rich_exception_info,
    split_manifest_lazy,
    split_sequence,
    suppress_and_warn,
)

Channels = Union[int, List[int]]


_DEFAULT_LHOTSE_AUDIO_DURATION_MISMATCH_TOLERANCE: Seconds = 0.025
LHOTSE_AUDIO_DURATION_MISMATCH_TOLERANCE: Seconds = (
    _DEFAULT_LHOTSE_AUDIO_DURATION_MISMATCH_TOLERANCE
)
FFMPEG_TORCHAUDIO_INFO_ENABLED: bool = True


def set_audio_duration_mismatch_tolerance(delta: Seconds) -> None:
    """
    Override Lhotse's global threshold for allowed audio duration mismatch between the
    manifest and the actual data.

    Some scenarios when a mismatch can happen:

        - the :class:`.Recording` manifest duration is rounded off too much
            (i.e., bad user input, but too inconvenient to go back and fix the manifests)

        - data augmentation changes the number of samples a bit in a difficult to predict way

    When there is a mismatch, Lhotse will either trim or replicate the diff to match
    the value found in the :class:`.Recording` manifest.

    .. note:: We don't recommend setting this lower than the default value, as it could
        break some data augmentation transforms.

    Example::

        >>> import lhotse
        >>> lhotse.set_audio_duration_mismatch_tolerance(0.01)  # 10ms tolerance

    :param delta: New tolerance in seconds.
    """
    global LHOTSE_AUDIO_DURATION_MISMATCH_TOLERANCE
    logging.info(
        "The user overrided tolerance for audio duration mismatch "
        "between the values in the manifest and the actual data. "
        f"Old threshold: {LHOTSE_AUDIO_DURATION_MISMATCH_TOLERANCE}s. "
        f"New threshold: {delta}s."
    )
    if delta < _DEFAULT_LHOTSE_AUDIO_DURATION_MISMATCH_TOLERANCE:
        warnings.warn(
            "The audio duration mismatch tolerance has been set to a value lower than "
            f"default ({_DEFAULT_LHOTSE_AUDIO_DURATION_MISMATCH_TOLERANCE}s). "
            f"We don't recommend this as it might break some data augmentation transforms."
        )
    LHOTSE_AUDIO_DURATION_MISMATCH_TOLERANCE = delta


def set_ffmpeg_torchaudio_info_enabled(enabled: bool) -> None:
    """
    Override Lhotse's global setting for whether to use ffmpeg-torchaudio to
    compute the duration of audio files. If disabled, we fall back to using a different
    backend such as sox_io or soundfile.

    .. note:: See this issue for more details: https://github.com/lhotse-speech/lhotse/issues/1026

    Example::

        >>> import lhotse
        >>> lhotse.set_ffmpeg_torchaudio_info_enabled(False)  # don't use ffmpeg-torchaudio

    :param enabled: Whether to use torchaudio to compute audio file duration.
    """
    global FFMPEG_TORCHAUDIO_INFO_ENABLED
    if enabled != FFMPEG_TORCHAUDIO_INFO_ENABLED:
        logging.info(
            "The user overrided the global setting for whether to use ffmpeg-torchaudio "
            "to compute the duration of audio files. "
            f"Old setting: {FFMPEG_TORCHAUDIO_INFO_ENABLED}. "
            f"New setting: {enabled}."
        )
    FFMPEG_TORCHAUDIO_INFO_ENABLED = enabled


# TODO: document the dataclasses like this:
# https://stackoverflow.com/a/3051356/5285891


@dataclass
class AudioSource:
    """
    AudioSource represents audio data that can be retrieved from somewhere.
    Supported sources of audio are currently:
    - 'file' (formats supported by soundfile, possibly multi-channel)
    - 'command' [unix pipe] (must be WAVE, possibly multi-channel)
    - 'url' (any URL type that is supported by "smart_open" library, e.g. http/https/s3/gcp/azure/etc.)
    - 'memory' (any format, read from a binary string attached to 'source' member of AudioSource)
    """

    type: str
    channels: List[int]
    source: Union[str, bytes]

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

        # TODO: refactor when another source type is added
        source = self.source

        if self.type == "command":
            if offset != 0.0 or duration is not None:
                # TODO(pzelasko): How should we support chunking for commands?
                #                 We risk being very inefficient when reading many chunks from the same file
                #                 without some caching scheme, because we'll be re-running commands.
                warnings.warn(
                    "You requested a subset of a recording that is read from disk via a bash command. "
                    "Expect large I/O overhead if you are going to read many chunks like these, "
                    "since every time we will read the whole file rather than its subset."
                )
            source = BytesIO(run(self.source, shell=True, stdout=PIPE).stdout)
            samples, sampling_rate = read_audio(
                source, offset=offset, duration=duration
            )

        elif self.type == "url":
            if offset != 0.0 or duration is not None:
                # TODO(pzelasko): How should we support chunking for URLs?
                #                 We risk being very inefficient when reading many chunks from the same file
                #                 without some caching scheme, because we'll be re-running commands.
                warnings.warn(
                    "You requested a subset of a recording that is read from URL. "
                    "Expect large I/O overhead if you are going to read many chunks like these, "
                    "since every time we will download the whole file rather than its subset."
                )
            with SmartOpen.open(self.source, "rb") as f:
                source = BytesIO(f.read())
                samples, sampling_rate = read_audio(
                    source, offset=offset, duration=duration
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
                available_duration < duration - LHOTSE_AUDIO_DURATION_MISMATCH_TOLERANCE
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


@dataclass
class Recording:
    """
    The :class:`~lhotse.audio.Recording` manifest describes the recordings in a given corpus.
    It contains information about the recording, such as its path(s), duration, the number of samples, etc.
    It allows to represent multiple channels coming from one or more files.

    This manifest does not specify any segmentation information or supervision such as the transcript or the speaker
    -- we use :class:`~lhotse.supervision.SupervisionSegment` for that.

    Note that :class:`~lhotse.audio.Recording` can represent both a single utterance (e.g., in LibriSpeech)
    and a 1-hour session with multiple channels and speakers (e.g., in AMI).
    In the latter case, it is partitioned into data suitable for model training using :class:`~lhotse.cut.Cut`.

    .. hint::
        Lhotse reads audio recordings using `pysoundfile`_ and `audioread`_, similarly to librosa,
        to support multiple audio formats. For OPUS files we require ffmpeg to be installed.

    .. hint::
        Since we support importing Kaldi data dirs, if ``wav.scp`` contains unix pipes,
        :class:`~lhotse.audio.Recording` will also handle them correctly.

    Examples

        A :class:`~lhotse.audio.Recording` can be simply created from a local audio file::

            >>> from lhotse import RecordingSet, Recording, AudioSource
            >>> recording = Recording.from_file('meeting.wav')
            >>> recording
            Recording(
                id='meeting',
                sources=[AudioSource(type='file', channels=[0], source='meeting.wav')],
                sampling_rate=16000,
                num_samples=57600000,
                duration=3600.0,
                transforms=None
            )

        This manifest can be easily converted to a Python dict and serialized to JSON/JSONL/YAML/etc::

            >>> recording.to_dict()
            {'id': 'meeting',
             'sources': [{'type': 'file',
               'channels': [0],
               'source': 'meeting.wav'}],
             'sampling_rate': 16000,
             'num_samples': 57600000,
             'duration': 3600.0}

        Recordings can be also created programatically, e.g. when they refer to URLs stored in S3 or somewhere else::

            >>> s3_audio_files = ['s3://my-bucket/123-5678.flac', ...]
            >>> recs = RecordingSet.from_recordings(
            ...     Recording(
            ...         id=url.split('/')[-1].replace('.flac', ''),
            ...         sources=[AudioSource(type='url', source=url, channels=[0])],
            ...         sampling_rate=16000,
            ...         num_samples=get_num_samples(url),
            ...         duration=get_duration(url)
            ...     )
            ...     for url in s3_audio_files
            ... )

        It allows reading a subset of the audio samples as a numpy array::

            >>> samples = recording.load_audio()
            >>> assert samples.shape == (1, 16000)
            >>> samples2 = recording.load_audio(offset=0.5)
            >>> assert samples2.shape == (1, 8000)
    """

    id: str
    sources: List[AudioSource]
    sampling_rate: int
    num_samples: int
    duration: Seconds
    channel_ids: Optional[List[int]] = None
    transforms: Optional[List[Dict]] = None

    def __post_init__(self):
        if self.channel_ids is None:
            self.channel_ids = sorted(
                cid for source in self.sources for cid in source.channels
            )

    @property
    def num_channels(self):
        return len(self.channel_ids)

    @staticmethod
    def from_file(
        path: Pathlike,
        recording_id: Optional[Union[str, Callable[[Path], str]]] = None,
        relative_path_depth: Optional[int] = None,
        force_opus_sampling_rate: Optional[int] = None,
        force_read_audio: bool = False,
    ) -> "Recording":
        """
        Read an audio file's header and create the corresponding ``Recording``.
        Suitable to use when each physical file represents a separate recording session.

        .. caution::
            If a recording session consists of multiple files (e.g. one per channel),
            it is advisable to create the ``Recording`` object manually, with each
            file represented as a separate ``AudioSource`` object.

        :param path: Path to an audio file supported by libsoundfile (pysoundfile).
        :param recording_id: recording id, when not specified ream the filename's stem ("x.wav" -> "x").
            It can be specified as a string or a function that takes the recording path and returns a string.
        :param relative_path_depth: optional int specifying how many last parts of the file path
            should be retained in the ``AudioSource``. By default writes the path as is.
        :param force_opus_sampling_rate: when specified, this value will be used as the sampling rate
            instead of the one we read from the manifest. This is useful for OPUS files that always
            have 48kHz rate and need to be resampled to the real one -- we will perform that operation
            "under-the-hood". For non-OPUS files this input is undefined.
        :param force_read_audio: Set it to ``True`` for audio files that do not have any metadata
            in their headers (e.g., "The People's Speech" FLAC files).
        :return: a new ``Recording`` instance pointing to the audio file.
        """
        path = Path(path)
        recording_id = (
            path.stem
            if recording_id is None
            else recording_id(path)
            if callable(recording_id)
            else recording_id
        )
        audio_info = info(
            path,
            force_opus_sampling_rate=force_opus_sampling_rate,
            force_read_audio=force_read_audio,
        )
        return Recording(
            id=recording_id,
            sampling_rate=audio_info.samplerate,
            num_samples=audio_info.frames,
            duration=audio_info.duration,
            sources=[
                AudioSource(
                    type="file",
                    channels=list(range(audio_info.channels)),
                    source=(
                        "/".join(path.parts[-relative_path_depth:])
                        if relative_path_depth is not None and relative_path_depth > 0
                        else str(path)
                    ),
                )
            ],
        )

    @staticmethod
    def from_bytes(
        data: bytes,
        recording_id: str,
    ) -> "Recording":
        """
        Like :meth:`.Recording.from_file`, but creates a manifest for a byte string with
        raw encoded audio data. This data is first decoded to obtain info such as the
        sampling rate, number of channels, etc. Then, the binary data is attached to the
        manifest. Calling :meth:`.Recording.load_audio` does not perform any I/O and
        instead decodes the byte string contents in memory.

        .. note:: Intended use of this method is for packing Recordings into archives
            where metadata and data should be available together
            (e.g., in WebDataset style tarballs).

        .. caution:: Manifest created with this method cannot be stored as JSON
            because JSON doesn't allow serializing binary data.

        :param data: bytes, byte string containing encoded audio contents.
        :param recording_id: recording id, unique string identifier.
        :return: a new ``Recording`` instance that owns the byte string data.
        """
        stream = BytesIO(data)
        audio_info = torchaudio_info(stream)
        return Recording(
            id=recording_id,
            sampling_rate=audio_info.samplerate,
            num_samples=audio_info.frames,
            duration=audio_info.duration,
            sources=[
                AudioSource(
                    type="memory",
                    channels=list(range(audio_info.channels)),
                    source=data,
                )
            ],
        )

    def move_to_memory(
        self,
        channels: Optional[Channels] = None,
        offset: Seconds = None,
        duration: Optional[Seconds] = None,
        format: Optional[str] = None,
    ) -> "Recording":
        """
        Read audio data and return a copy of the manifest with binary data attached.
        Calling :meth:`.Recording.load_audio` on that copy will not trigger I/O.

        If all arguments are left as defaults, we won't decode the audio and attach
        the bytes we read from disk/other source as-is.
        If ``channels``, ``duration``, or ``offset`` are specified, we'll decode the
        audio and re-encode it into ``format`` before attaching.
        The default format is FLAC, other formats compatible with torchaudio.save are
        also accepted.
        """

        if all(src.type == "memory" for src in self.sources):
            return self  # nothing to do

        # Case #1: no opts specified, read audio without decoding and move it in memory.
        if all(opt is None for opt in (channels, offset, duration)):
            memory_sources = [
                AudioSource(
                    type="memory",
                    channels=old_source.channels,
                    source=open(old_source.source, "rb").read(),
                )
                for old_source in self.sources
            ]
            return fastcopy(self, sources=memory_sources)

        # Case #2: user specified some subset of the recording, decode audio,
        #          subset it, and encode it again but save in memory.
        import torchaudio

        audio = self.load_audio(
            channels=channels, offset=ifnone(offset, 0), duration=duration
        )
        stream = BytesIO()
        torchaudio_save_flac_safe(
            stream, torch.from_numpy(audio), self.sampling_rate, format=format
        )
        channels = (ifnone(channels, self.channel_ids),)
        if isinstance(channels, int):
            channels = [channels]
        return Recording(
            id=self.id,
            sources=[
                AudioSource(
                    type="memory",
                    channels=channels,
                    source=stream.getvalue(),
                )
            ],
            sampling_rate=self.sampling_rate,
            num_samples=audio.shape[1],
            duration=ifnone(duration, self.duration),
        )

    def to_dict(self) -> dict:
        return asdict_nonull(self)

    def to_cut(self):
        """
        Create a Cut out of this recording --- MonoCut or MultiCut, depending on the
        number of channels.
        """
        from lhotse.cut import MonoCut, MultiCut

        cls = MonoCut if self.num_channels == 1 else MultiCut
        return cls(
            id=self.id,
            start=0.0,
            duration=self.duration,
            channel=self.channel_ids[0] if self.num_channels == 1 else self.channel_ids,
            recording=self,
        )

    @rich_exception_info
    def load_audio(
        self,
        channels: Optional[Channels] = None,
        offset: Seconds = 0.0,
        duration: Optional[Seconds] = None,
    ) -> np.ndarray:
        """
        Read the audio samples from the underlying audio source (path, URL, unix pipe/command).

        :param channels: int or iterable of ints, a subset of channel IDs to read (reads all by default).
        :param offset: seconds, where to start reading the audio (at offset 0 by default).
            Note that it is only efficient for local filesystem files, i.e. URLs and commands will read
            all the samples first and discard the unneeded ones afterwards.
        :param duration: seconds, indicates the total audio time to read (starting from ``offset``).
        :return: a numpy array of audio samples with shape ``(num_channels, num_samples)``.
        """

        assert offset <= self.duration, (
            f"Cannot load audio because the Recording's duration {self.duration}s "
            f"is smaller than the requested offset {offset}s."
        )

        if channels is None:
            channels = SetContainingAnything()
        else:
            channels = frozenset([channels] if isinstance(channels, int) else channels)
            recording_channels = frozenset(self.channel_ids)
            assert channels.issubset(recording_channels), (
                "Requested to load audio from a channel "
                "that does not exist in the recording: "
                f"(recording channels: {recording_channels} -- "
                f"requested channels: {channels})"
            )

        transforms = [
            AudioTransform.from_dict(params) for params in self.transforms or []
        ]

        # Do a "backward pass" over data augmentation transforms to get the
        # offset and duration for loading a piece of the original audio.
        offset_aug, duration_aug = offset, duration
        for tfn in reversed(transforms):
            offset_aug, duration_aug = tfn.reverse_timestamps(
                offset=offset_aug,
                duration=duration_aug,
                sampling_rate=self.sampling_rate,
            )

        samples_per_source = []
        for source in self.sources:
            # Case: source not requested
            if not channels.intersection(source.channels):
                continue
            samples = source.load_audio(
                offset=offset_aug,
                duration=duration_aug,
                force_opus_sampling_rate=self.sampling_rate,
            )

            # Case: two-channel audio file but only one channel requested
            #       it might not be optimal to load all channels, but IDK if there's anything we can do about it
            channels_to_remove = [
                idx for idx, cid in enumerate(source.channels) if cid not in channels
            ]
            if channels_to_remove:
                samples = np.delete(samples, channels_to_remove, axis=0)
            samples_per_source.append(samples)

        # Stack all the samples from all the sources into a single array.
        audio = self._stack_audio_channels(samples_per_source)

        # We'll apply the transforms now (if any).
        for tfn in transforms:
            audio = tfn(audio, self.sampling_rate)

        # Transformation chains can introduce small mismatches in the number of samples:
        # we'll fix them here, or raise an error if they exceeded a tolerance threshold.
        audio = assert_and_maybe_fix_num_samples(
            audio, offset=offset, duration=duration, recording=self
        )

        return audio

    def _stack_audio_channels(self, samples_per_source: List[np.ndarray]) -> np.ndarray:
        # There may be a mismatch in the number of samples between different channels. We
        # check if the mismatch is within a reasonable tolerance and if so, we pad
        # all channels to the length of the longest one.
        allowed_diff = int(
            compute_num_samples(
                LHOTSE_AUDIO_DURATION_MISMATCH_TOLERANCE,
                sampling_rate=self.sampling_rate,
            )
        )
        if len(samples_per_source) > 1:
            # Make all arrays 2D
            samples_per_source = [
                s[None, :] if s.ndim == 1 else s for s in samples_per_source
            ]
            max_samples = max(s.shape[1] for s in samples_per_source)
            for i, s in enumerate(samples_per_source):
                if max_samples - s.shape[1] <= allowed_diff:
                    s = np.pad(s, ((0, 0), (0, max_samples - s.shape[1])), "constant")
                    samples_per_source[i] = s
                else:
                    raise DurationMismatchError(
                        f"The mismatch between the number of samples in the "
                        f"different channels of the recording {self.id} is "
                        f"greater than the allowed tolerance {LHOTSE_AUDIO_DURATION_MISMATCH_TOLERANCE}."
                    )
            audio = np.concatenate(samples_per_source, axis=0)
        else:
            # shape: (n_channels, n_samples)
            audio = np.vstack(samples_per_source)
        return audio

    def _expected_num_samples(
        self, offset: Seconds, duration: Optional[Seconds]
    ) -> int:
        if offset == 0 and duration is None:
            return self.num_samples
        duration = duration if duration is not None else self.duration - offset
        return compute_num_samples(duration, sampling_rate=self.sampling_rate)

    def with_path_prefix(self, path: Pathlike) -> "Recording":
        return fastcopy(self, sources=[s.with_path_prefix(path) for s in self.sources])

    def perturb_speed(self, factor: float, affix_id: bool = True) -> "Recording":
        """
        Return a new ``Recording`` that will lazily perturb the speed while loading audio.
        The ``num_samples`` and ``duration`` fields are updated to reflect the
        shrinking/extending effect of speed.

        :param factor: The speed will be adjusted this many times (e.g. factor=1.1 means 1.1x faster).
        :param affix_id: When true, we will modify the ``Recording.id`` field
            by affixing it with "_sp{factor}".
        :return: a modified copy of the current ``Recording``.
        """
        transforms = self.transforms.copy() if self.transforms is not None else []
        transforms.append(Speed(factor=factor).to_dict())
        new_num_samples = perturb_num_samples(self.num_samples, factor)
        new_duration = new_num_samples / self.sampling_rate
        return fastcopy(
            self,
            id=f"{self.id}_sp{factor}" if affix_id else self.id,
            num_samples=new_num_samples,
            duration=new_duration,
            transforms=transforms,
        )

    def perturb_tempo(self, factor: float, affix_id: bool = True) -> "Recording":
        """
        Return a new ``Recording`` that will lazily perturb the tempo while loading audio.

        Compared to speed perturbation, tempo preserves pitch.
        The ``num_samples`` and ``duration`` fields are updated to reflect the
        shrinking/extending effect of tempo.

        :param factor: The tempo will be adjusted this many times (e.g. factor=1.1 means 1.1x faster).
        :param affix_id: When true, we will modify the ``Recording.id`` field
            by affixing it with "_tp{factor}".
        :return: a modified copy of the current ``Recording``.
        """
        transforms = self.transforms.copy() if self.transforms is not None else []
        transforms.append(Tempo(factor=factor).to_dict())
        new_num_samples = perturb_num_samples(self.num_samples, factor)
        new_duration = new_num_samples / self.sampling_rate
        return fastcopy(
            self,
            id=f"{self.id}_tp{factor}" if affix_id else self.id,
            num_samples=new_num_samples,
            duration=new_duration,
            transforms=transforms,
        )

    def perturb_volume(self, factor: float, affix_id: bool = True) -> "Recording":
        """
        Return a new ``Recording`` that will lazily perturb the volume while loading audio.

        :param factor: The volume scale to be applied (e.g. factor=1.1 means 1.1x louder).
        :param affix_id: When true, we will modify the ``Recording.id`` field
            by affixing it with "_tp{factor}".
        :return: a modified copy of the current ``Recording``.
        """
        transforms = self.transforms.copy() if self.transforms is not None else []
        transforms.append(Volume(factor=factor).to_dict())
        return fastcopy(
            self,
            id=f"{self.id}_vp{factor}" if affix_id else self.id,
            transforms=transforms,
        )

    def normalize_loudness(self, target: float, affix_id: bool = False) -> "Recording":
        """
        Return a new ``Recording`` that will lazily apply WPE dereverberation.

        :param target: The target loudness (in dB) to normalize to.
        :param affix_id: When true, we will modify the ``Recording.id`` field
            by affixing it with "_ln{factor}".
        :return: a modified copy of the current ``Recording``.
        """
        transforms = self.transforms.copy() if self.transforms is not None else []
        transforms.append(LoudnessNormalization(target=target).to_dict())
        return fastcopy(
            self,
            id=f"{self.id}_ln{target}" if affix_id else self.id,
            transforms=transforms,
        )

    def dereverb_wpe(self, affix_id: bool = True) -> "Recording":
        """
        Return a new ``Recording`` that will lazily apply WPE dereverberation.

        :param affix_id: When true, we will modify the ``Recording.id`` field
            by affixing it with "_wpe".
        :return: a modified copy of the current ``Recording``.
        """
        transforms = self.transforms.copy() if self.transforms is not None else []
        transforms.append(DereverbWPE().to_dict())
        return fastcopy(
            self,
            id=f"{self.id}_wpe" if affix_id else self.id,
            transforms=transforms,
        )

    def reverb_rir(
        self,
        rir_recording: Optional["Recording"] = None,
        normalize_output: bool = True,
        early_only: bool = False,
        affix_id: bool = True,
        rir_channels: Optional[List[int]] = None,
        room_rng_seed: Optional[int] = None,
        source_rng_seed: Optional[int] = None,
    ) -> "Recording":
        """
        Return a new ``Recording`` that will lazily apply reverberation based on provided
        impulse response while loading audio. If no impulse response is provided, we will
        generate an RIR using a fast random generator (https://arxiv.org/abs/2208.04101).

        :param rir_recording: The impulse response to be used.
        :param normalize_output: When true, output will be normalized to have energy as input.
        :param early_only: When true, only the early reflections (first 50 ms) will be used.
        :param affix_id: When true, we will modify the ``Recording.id`` field
            by affixing it with "_rvb".
        :param rir_channels: The channels of the impulse response to be used (in case of multi-channel
            impulse responses). By default, only the first channel is used. If no RIR is
            provided, we will generate one with as many channels as this argument specifies.
        :param room_rng_seed: The seed to be used for the room configuration.
        :param source_rng_seed: The seed to be used for the source position.
        :return: the perturbed ``Recording``.
        """
        if rir_recording is not None:
            assert (
                rir_recording.sampling_rate == self.sampling_rate
            ), f"Sampling rate mismatch between RIR vs recording: {rir_recording.sampling_rate} vs {self.sampling_rate}."

        # We may need to change the `channel_ids` field according to whether we are convolving
        # with a multi-channel RIR or not.
        # The following cases are possible:
        # Case 1: input is mono, rir is mono -> mono output, no need to change
        # Case 2: input is mono, rir is multi-channel -> multi-channel output, change channel_ids
        # Case 3: input is multi-channel, rir is mono -> multi-channel output, no need to change
        # Case 4: input is multi-channel, rir is multi-channel -> multi-channel output,
        #   no need to change (since we assume that the RIR has the same number of channels as the input)

        if self.num_channels > 1 or rir_channels is None or len(rir_channels) == 1:
            # Case 1, 3 or 4
            new_channel_ids = self.channel_ids
        else:
            # Case 2
            new_channel_ids = list(range(len(rir_channels)))

        if rir_recording is None:
            from lhotse.augmentation.utils import FastRandomRIRGenerator

            rir_generator = FastRandomRIRGenerator(
                sr=self.sampling_rate,
                room_seed=room_rng_seed,
                source_seed=source_rng_seed,
            )
        else:
            rir_generator = None

        transforms = self.transforms.copy() if self.transforms is not None else []
        transforms.append(
            ReverbWithImpulseResponse(
                rir=rir_recording,
                normalize_output=normalize_output,
                early_only=early_only,
                rir_channels=rir_channels if rir_channels is not None else [0],
                rir_generator=rir_generator,
            ).to_dict()
        )
        return fastcopy(
            self,
            id=f"{self.id}_rvb" if affix_id else self.id,
            channel_ids=new_channel_ids,
            transforms=transforms,
        )

    def resample(self, sampling_rate: int) -> "Recording":
        """
        Return a new ``Recording`` that will be lazily resampled while loading audio.
        :param sampling_rate: The new sampling rate.
        :return: A resampled ``Recording``.
        """
        if sampling_rate == self.sampling_rate:
            return fastcopy(self)

        transforms = self.transforms.copy() if self.transforms is not None else []

        if not any(
            isinstance(s.source, str) and s.source.endswith(".opus")
            for s in self.sources
        ):
            # OPUS is a special case for resampling.
            # Normally, we use Torchaudio SoX bindings for resampling,
            # but in case of OPUS we ask FFMPEG to resample it during
            # decoding as its faster.
            # Because of that, we have to skip adding a transform
            # for OPUS files and only update the metadata in the manifest.
            transforms.append(
                Resample(
                    source_sampling_rate=self.sampling_rate,
                    target_sampling_rate=sampling_rate,
                ).to_dict()
            )

        new_num_samples = compute_num_samples(
            self.duration, sampling_rate, rounding=ROUND_HALF_UP
        )
        # Duration might need an adjustment when doing a non-trivial resampling
        # (e.g. 16000 -> 22050), where the resulting number of samples cannot
        # correspond to old duration exactly.
        new_duration = new_num_samples / sampling_rate
        return fastcopy(
            self,
            duration=new_duration,
            num_samples=new_num_samples,
            sampling_rate=sampling_rate,
            transforms=transforms,
        )

    @staticmethod
    def from_dict(data: dict) -> "Recording":
        raw_sources = data.pop("sources")
        return Recording(
            sources=[AudioSource.from_dict(s) for s in raw_sources], **data
        )


class RecordingSet(Serializable, AlgorithmMixin):
    """
    :class:`~lhotse.audio.RecordingSet` represents a collection of recordings, indexed by recording IDs.
    It does not contain any annotation such as the transcript or the speaker identity --
    just the information needed to retrieve a recording such as its path, URL, number of channels,
    and some recording metadata (duration, number of samples).

    It also supports (de)serialization to/from YAML/JSON/etc. and takes care of mapping between
    rich Python classes and YAML/JSON/etc. primitives during conversion.

    When coming from Kaldi, think of it as ``wav.scp`` on steroids: :class:`~lhotse.audio.RecordingSet`
    also has the information from *reco2dur* and *reco2num_samples*,
    is able to represent multi-channel recordings and read a specified subset of channels,
    and support reading audio files directly, via a unix pipe, or downloading them on-the-fly from a URL
    (HTTPS/S3/Azure/GCP/etc.).

    Examples:

        :class:`~lhotse.audio.RecordingSet` can be created from an iterable of :class:`~lhotse.audio.Recording` objects::

            >>> from lhotse import RecordingSet
            >>> audio_paths = ['123-5678.wav', ...]
            >>> recs = RecordingSet.from_recordings(Recording.from_file(p) for p in audio_paths)

        As well as from a directory, which will be scanned recursively for files with parallel processing::

            >>> recs2 = RecordingSet.from_dir('/data/audio', pattern='*.flac', num_jobs=4)

        It behaves similarly to a ``dict``::

            >>> '123-5678' in recs
            True
            >>> recording = recs['123-5678']
            >>> for recording in recs:
            >>>    pass
            >>> len(recs)
            127

        It also provides some utilities for I/O::

            >>> recs.to_file('recordings.jsonl')
            >>> recs.to_file('recordings.json.gz')  # auto-compression
            >>> recs2 = RecordingSet.from_file('recordings.jsonl')

        Manipulation::

            >>> longer_than_5s = recs.filter(lambda r: r.duration > 5)
            >>> first_100 = recs.subset(first=100)
            >>> split_into_4 = recs.split(num_splits=4)
            >>> shuffled = recs.shuffle()

        And lazy data augmentation/transformation, that requires to adjust some information
        in the manifest (e.g., ``num_samples`` or ``duration``).
        Note that in the following examples, the audio is untouched -- the operations are stored in the manifest,
        and executed upon reading the audio::

            >>> recs_sp = recs.perturb_speed(factor=1.1)
            >>> recs_vp = recs.perturb_volume(factor=2.)
            >>> recs_rvb = recs.reverb_rir(rir_recs)
            >>> recs_24k = recs.resample(24000)
    """

    def __init__(self, recordings: Optional[Mapping[str, Recording]] = None) -> None:
        self.recordings = ifnone(recordings, {})

    def __eq__(self, other: "RecordingSet") -> bool:
        return self.recordings == other.recordings

    @property
    def data(self) -> Union[Dict[str, Recording], Iterable[Recording]]:
        """Alias property for ``self.recordings``"""
        return self.recordings

    @property
    def ids(self) -> Iterable[str]:
        return self.recordings.keys()

    @staticmethod
    def from_recordings(recordings: Iterable[Recording]) -> "RecordingSet":
        return RecordingSet(recordings=index_by_id_and_check(recordings))

    from_items = from_recordings

    @staticmethod
    def from_dir(
        path: Pathlike,
        pattern: str,
        num_jobs: int = 1,
        force_opus_sampling_rate: Optional[int] = None,
        recording_id: Optional[Callable[[Path], str]] = None,
    ):
        """
        Recursively scan a directory ``path`` for audio files that match the given ``pattern`` and create
        a :class:`.RecordingSet` manifest for them.
        Suitable to use when each physical file represents a separate recording session.

        .. caution::
            If a recording session consists of multiple files (e.g. one per channel),
            it is advisable to create each :class:`.Recording` object manually, with each
            file represented as a separate :class:`.AudioSource` object, and then
            a :class:`RecordingSet` that contains all the recordings.

        :param path: Path to a directory of audio of files (possibly with sub-directories).
        :param pattern: A bash-like pattern specifying allowed filenames, e.g. ``*.wav`` or ``session1-*.flac``.
        :param num_jobs: The number of parallel workers for reading audio files to get their metadata.
        :param force_opus_sampling_rate: when specified, this value will be used as the sampling rate
            instead of the one we read from the manifest. This is useful for OPUS files that always
            have 48kHz rate and need to be resampled to the real one -- we will perform that operation
            "under-the-hood". For non-OPUS files this input does nothing.
        :param recording_id: A function which takes the audio file path and returns the recording ID. If not
            specified, the filename will be used as the recording ID.
        :return: a new ``Recording`` instance pointing to the audio file.
        """
        msg = f"Scanning audio files ({pattern})"

        file_read_worker = partial(
            Recording.from_file,
            force_opus_sampling_rate=force_opus_sampling_rate,
            recording_id=recording_id,
        )

        if num_jobs == 1:
            # Avoid spawning process for one job.
            return RecordingSet.from_recordings(
                tqdm(map(file_read_worker, Path(path).rglob(pattern)), desc=msg)
            )
        with ProcessPoolExecutor(num_jobs) as ex:
            return RecordingSet.from_recordings(
                tqdm(
                    ex.map(file_read_worker, Path(path).rglob(pattern)),
                    desc=msg,
                )
            )

    @staticmethod
    def from_dicts(data: Iterable[dict]) -> "RecordingSet":
        return RecordingSet.from_recordings(
            Recording.from_dict(raw_rec) for raw_rec in data
        )

    def to_dicts(self) -> Iterable[dict]:
        return (r.to_dict() for r in self)

    def split(
        self, num_splits: int, shuffle: bool = False, drop_last: bool = False
    ) -> List["RecordingSet"]:
        """
        Split the :class:`~lhotse.RecordingSet` into ``num_splits`` pieces of equal size.

        :param num_splits: Requested number of splits.
        :param shuffle: Optionally shuffle the recordings order first.
        :param drop_last: determines how to handle splitting when ``len(seq)`` is not divisible
            by ``num_splits``. When ``False`` (default), the splits might have unequal lengths.
            When ``True``, it may discard the last element in some splits to ensure they are
            equally long.
        :return: A list of :class:`~lhotse.RecordingSet` pieces.
        """
        return [
            RecordingSet.from_recordings(subset)
            for subset in split_sequence(
                self, num_splits=num_splits, shuffle=shuffle, drop_last=drop_last
            )
        ]

    def split_lazy(
        self, output_dir: Pathlike, chunk_size: int, prefix: str = ""
    ) -> List["RecordingSet"]:
        """
        Splits a manifest (either lazily or eagerly opened) into chunks, each
        with ``chunk_size`` items (except for the last one, typically).

        In order to be memory efficient, this implementation saves each chunk
        to disk in a ``.jsonl.gz`` format as the input manifest is sampled.

        .. note:: For lowest memory usage, use ``load_manifest_lazy`` to open the
            input manifest for this method.

        :param output_dir: directory where the split manifests are saved.
            Each manifest is saved at: ``{output_dir}/{prefix}.{split_idx}.jsonl.gz``
        :param chunk_size: the number of items in each chunk.
        :param prefix: the prefix of each manifest.
        :return: a list of lazily opened chunk manifests.
        """
        return split_manifest_lazy(
            self, output_dir=output_dir, chunk_size=chunk_size, prefix=prefix
        )

    def subset(
        self, first: Optional[int] = None, last: Optional[int] = None
    ) -> "RecordingSet":
        """
        Return a new ``RecordingSet`` according to the selected subset criterion.
        Only a single argument to ``subset`` is supported at this time.

        :param first: int, the number of first recordings to keep.
        :param last: int, the number of last recordings to keep.
        :return: a new ``RecordingSet`` with the subset results.
        """
        assert exactly_one_not_null(
            first, last
        ), "subset() can handle only one non-None arg."

        if first is not None:
            assert first > 0
            out = RecordingSet.from_items(islice(self, first))
            if len(out) < first:
                logging.warning(
                    f"RecordingSet has only {len(out)} items but first {first} were requested."
                )
            return out

        if last is not None:
            assert last > 0
            if last > len(self):
                logging.warning(
                    f"RecordingSet has only {len(self)} items but last {last} required; "
                    f"not doing anything."
                )
                return self
            return RecordingSet.from_recordings(
                islice(self, len(self) - last, len(self))
            )

    def load_audio(
        self,
        recording_id: str,
        channels: Optional[Channels] = None,
        offset_seconds: float = 0.0,
        duration_seconds: Optional[float] = None,
    ) -> np.ndarray:
        return self.recordings[recording_id].load_audio(
            channels=channels, offset=offset_seconds, duration=duration_seconds
        )

    def with_path_prefix(self, path: Pathlike) -> "RecordingSet":
        return RecordingSet.from_recordings(r.with_path_prefix(path) for r in self)

    def num_channels(self, recording_id: str) -> int:
        return self.recordings[recording_id].num_channels

    def sampling_rate(self, recording_id: str) -> int:
        return self.recordings[recording_id].sampling_rate

    def num_samples(self, recording_id: str) -> int:
        return self.recordings[recording_id].num_samples

    def duration(self, recording_id: str) -> Seconds:
        return self.recordings[recording_id].duration

    def perturb_speed(self, factor: float, affix_id: bool = True) -> "RecordingSet":
        """
        Return a new ``RecordingSet`` that will lazily perturb the speed while loading audio.
        The ``num_samples`` and ``duration`` fields are updated to reflect the
        shrinking/extending effect of speed.

        :param factor: The speed will be adjusted this many times (e.g. factor=1.1 means 1.1x faster).
        :param affix_id: When true, we will modify the ``Recording.id`` field
            by affixing it with "_sp{factor}".
        :return: a ``RecordingSet`` containing the perturbed ``Recording`` objects.
        """
        return RecordingSet.from_recordings(
            r.perturb_speed(factor=factor, affix_id=affix_id) for r in self
        )

    def perturb_tempo(self, factor: float, affix_id: bool = True) -> "RecordingSet":
        """
        Return a new ``RecordingSet`` that will lazily perturb the tempo while loading audio.
        The ``num_samples`` and ``duration`` fields are updated to reflect the
        shrinking/extending effect of tempo.

        :param factor: The speed will be adjusted this many times (e.g. factor=1.1 means 1.1x faster).
        :param affix_id: When true, we will modify the ``Recording.id`` field
            by affixing it with "_sp{factor}".
        :return: a ``RecordingSet`` containing the perturbed ``Recording`` objects.
        """
        return RecordingSet.from_recordings(
            r.perturb_tempo(factor=factor, affix_id=affix_id) for r in self
        )

    def perturb_volume(self, factor: float, affix_id: bool = True) -> "RecordingSet":
        """
        Return a new ``RecordingSet`` that will lazily perturb the volume while loading audio.

        :param factor: The volume scale to be applied (e.g. factor=1.1 means 1.1x louder).
        :param affix_id: When true, we will modify the ``Recording.id`` field
            by affixing it with "_sp{factor}".
        :return: a ``RecordingSet`` containing the perturbed ``Recording`` objects.
        """
        return RecordingSet.from_recordings(
            r.perturb_volume(factor=factor, affix_id=affix_id) for r in self
        )

    def reverb_rir(
        self,
        rir_recordings: Optional["RecordingSet"] = None,
        normalize_output: bool = True,
        early_only: bool = False,
        affix_id: bool = True,
        rir_channels: List[int] = [0],
        room_rng_seed: Optional[int] = None,
        source_rng_seed: Optional[int] = None,
    ) -> "RecordingSet":
        """
        Return a new ``RecordingSet`` that will lazily apply reverberation based on provided
        impulse responses while loading audio. If no ``rir_recordings`` are provided, we will
        generate a set of impulse responses using a fast random generator (https://arxiv.org/abs/2208.04101).

        :param rir_recordings: The impulse responses to be used.
        :param normalize_output: When true, output will be normalized to have energy as input.
        :param early_only: When true, only the early reflections (first 50 ms) will be used.
        :param affix_id: When true, we will modify the ``Recording.id`` field
            by affixing it with "_rvb".
        :param rir_channels: The channels to be used for the RIRs (if multi-channel). Uses first
            channel by default. If no RIR is provided, we will generate one with as many channels
            as this argument specifies.
        :param room_rng_seed: The seed to be used for the room configuration.
        :param source_rng_seed: The seed to be used for the source positions.
        :return: a ``RecordingSet`` containing the perturbed ``Recording`` objects.
        """
        rir_recordings = list(rir_recordings)
        return RecordingSet.from_recordings(
            r.reverb_rir(
                rir_recording=random.choice(rir_recordings) if rir_recordings else None,
                normalize_output=normalize_output,
                early_only=early_only,
                affix_id=affix_id,
                rir_channels=rir_channels,
                room_rng_seed=room_rng_seed,
                source_rng_seed=source_rng_seed,
            )
            for r in self
        )

    def resample(self, sampling_rate: int) -> "RecordingSet":
        """
        Apply resampling to all recordings in the ``RecordingSet`` and return a new ``RecordingSet``.
        :param sampling_rate: The new sampling rate.
        :return: a new ``RecordingSet`` with lazily resampled ``Recording`` objects.
        """
        return RecordingSet.from_recordings(r.resample(sampling_rate) for r in self)

    def __repr__(self) -> str:
        return f"RecordingSet(len={len(self)})"

    def __contains__(self, item: Union[str, Recording]) -> bool:
        if isinstance(item, str):
            return item in self.recordings
        else:
            return item.id in self.recordings

    def __getitem__(self, recording_id_or_index: Union[int, str]) -> Recording:
        if isinstance(recording_id_or_index, str):
            return self.recordings[recording_id_or_index]
        # ~100x faster than list(dict.values())[index] for 100k elements
        return next(
            val
            for idx, val in enumerate(self.recordings.values())
            if idx == recording_id_or_index
        )

    def __iter__(self) -> Iterable[Recording]:
        return iter(self.recordings.values())

    def __len__(self) -> int:
        return len(self.recordings)


class AudioMixer:
    """
    Utility class to mix multiple waveforms into a single one.
    It should be instantiated separately for each mixing session (i.e. each ``MixedCut``
    will create a separate ``AudioMixer`` to mix its tracks).
    It is initialized with a numpy array of audio samples (typically float32 in [-1, 1] range)
    that represents the "reference" signal for the mix.
    Other signals can be mixed to it with different time offsets and SNRs using the
    ``add_to_mix`` method.
    The time offset is relative to the start of the reference signal
    (only positive values are supported).
    The SNR is relative to the energy of the signal used to initialize the ``AudioMixer``.

    .. note:: Both single-channel and multi-channel signals are supported as reference
        and added signals. The only requirement is that the when mixing 2 multi-channel
        signals, they must have the same number of channels.

    .. note:: When the AudioMixer contains multi-channel tracks, 2 types of mixed signals
        can be generated:
        - `mixed_audio` mixes each channel independently, and returns a multi-channel signal.
          If there is a mono track, it is added to all the channels.
        - `mixed_mono_audio` mixes all channels together, and returns a single-channel signal.
    """

    def __init__(
        self,
        base_audio: np.ndarray,
        sampling_rate: int,
        reference_energy: Optional[float] = None,
    ):
        """
        AudioMixer's constructor.

        :param base_audio: A numpy array with the audio samples for the base signal
            (all the other signals will be mixed to it).
        :param sampling_rate: Sampling rate of the audio.
        :param reference_energy: Optionally pass a reference energy value to compute SNRs against.
            This might be required when ``base_audio`` corresponds to zero-padding.
        """
        self.tracks = [base_audio]
        self.offsets = [0]
        self.sampling_rate = sampling_rate
        self.num_channels = base_audio.shape[0]
        self.dtype = self.tracks[0].dtype

        # Keep a pre-computed energy value of the audio that we initialize the Mixer with;
        # it is required to compute gain ratios that satisfy SNR during the mix.
        if reference_energy is None:
            self.reference_energy = audio_energy(base_audio)
        else:
            self.reference_energy = reference_energy

    def _pad_track(
        self, audio: np.ndarray, offset: int, total: Optional[int] = None
    ) -> np.ndarray:
        assert audio.ndim == 2, f"audio.ndim={audio.ndim}"
        if total is None:
            total = audio.shape[1] + offset
        assert (
            audio.shape[1] + offset <= total
        ), f"{audio.shape[1]} + {offset} <= {total}"
        return np.pad(
            audio, pad_width=((0, 0), (offset, total - audio.shape[1] - offset))
        )

    @property
    def num_samples_total(self) -> int:
        longest = 0
        for offset, audio in zip(self.offsets, self.tracks):
            longest = max(longest, offset + audio.shape[1])
        return longest

    @property
    def unmixed_audio(self) -> List[np.ndarray]:
        """
        Return a list of numpy arrays with the shape (C, num_samples), where each track is
        zero padded and scaled adequately to the offsets and SNR used in ``add_to_mix`` call.
        """
        total = self.num_samples_total
        return [
            self._pad_track(track, offset=offset, total=total)
            for offset, track in zip(self.offsets, self.tracks)
        ]

    @property
    def mixed_audio(self) -> np.ndarray:
        """
        Return a numpy ndarray with the shape (num_channels, num_samples) - a mix of the tracks
        supplied with ``add_to_mix`` calls.
        """
        total = self.num_samples_total
        mixed = np.zeros((self.num_channels, total), dtype=self.dtype)
        for offset, track in zip(self.offsets, self.tracks):
            # Only two cases are possible here: either the track is mono, or it has the same
            # number of channels as the mixer. For the latter case, we don't need to do anything
            # special, as we can just add the track to the mix. For the former case, we need to
            # add the mono track to all channels by repeating it.
            if track.shape[0] == 1 and self.num_channels > 1:
                track = np.tile(track, (self.num_channels, 1))
            mixed[:, offset : offset + track.shape[1]] += track
        return mixed

    @property
    def mixed_mono_audio(self) -> np.ndarray:
        """
        Return a numpy ndarray with the shape (1, num_samples) - a mix of the tracks
        supplied with ``add_to_mix`` calls.
        """
        total = self.num_samples_total
        mixed = np.zeros((1, total), dtype=self.dtype)
        for offset, track in zip(self.offsets, self.tracks):
            if track.shape[0] > 1:
                # Sum all channels of the track
                track = np.sum(track, axis=0, keepdims=True)
            mixed[:, offset : offset + track.shape[1]] += track
        return mixed

    def add_to_mix(
        self,
        audio: np.ndarray,
        snr: Optional[Decibels] = None,
        offset: Seconds = 0.0,
    ):
        """
        Add audio of a new track into the mix.
        :param audio: An array of audio samples to be mixed in.
        :param snr: Signal-to-noise ratio, assuming `audio` represents noise (positive SNR - lower `audio` energy,
        negative SNR - higher `audio` energy)
        :param offset: How many seconds to shift `audio` in time. For mixing, the signal will be padded before
        the start with low energy values.
        :return:
        """
        if audio.size == 0:
            return  # do nothing for empty arrays

        assert offset >= 0.0, "Negative offset in mixing is not supported."

        num_samples_offset = compute_num_samples(offset, self.sampling_rate)

        # When SNR is requested, find what gain is needed to satisfy the SNR
        gain = 1.0
        if snr is not None and self.reference_energy > 0:
            added_audio_energy = audio_energy(audio)
            if added_audio_energy > 0.0:
                target_energy = self.reference_energy * (10.0 ** (-snr / 10))
                # When mixing time-domain signals, we are working with root-power (field) quantities,
                # whereas the energy ratio applies to power quantities. To compute the gain correctly,
                # we need to take a square root of the energy ratio.
                gain = sqrt(target_energy / added_audio_energy)
        self.tracks.append(gain * audio)
        self.offsets.append(num_samples_offset)
        # We cannot mix 2 multi-channel audios with different number of channels.
        if (
            audio.shape[0] != self.num_channels
            and self.num_channels != 1
            and audio.shape[0] != 1
        ):
            raise ValueError(
                f"Cannot mix audios with {audio.shape[0]} and {self.num_channels} channels."
            )
        self.num_channels = max(self.num_channels, audio.shape[0])


def audio_energy(audio: np.ndarray) -> float:
    return float(np.average(audio**2))


FileObject = Any  # Alias for file-like objects


@dynamic_lru_cache
def read_audio(
    path_or_fd: Union[Pathlike, FileObject],
    offset: Seconds = 0.0,
    duration: Optional[Seconds] = None,
    force_opus_sampling_rate: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    return get_default_audio_backend().read_audio(
        path_or_fd=path_or_fd,
        offset=offset,
        duration=duration,
        force_opus_sampling_rate=force_opus_sampling_rate,
    )


class AudioBackend:
    """
    Internal Lhotse abstraction. An AudioBackend defines three methods:
    one for reading audio, and two filters that help determine if it should be used.

    ``handle_special_case`` means this backend should be exclusively
    used for a given type of input path/file.

    ``is_applicable`` means this backend most likely can be used for a given type of input path/file,
    but it may also fail. Its purpose is more to filter out formats that definitely are not supported.
    """

    def read_audio(
        self,
        path_or_fd: Union[Pathlike, FileObject],
        offset: Seconds = 0.0,
        duration: Optional[Seconds] = None,
        force_opus_sampling_rate: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        raise NotImplementedError()

    def handles_special_case(self, path_or_fd: Union[Pathlike, FileObject]) -> bool:
        return False

    def is_applicable(self, path_or_fd: Union[Pathlike, FileObject]) -> bool:
        return True


class FfmpegSubprocessOpusBackend(AudioBackend):
    def read_audio(
        self,
        path_or_fd: Union[Pathlike, FileObject],
        offset: Seconds = 0.0,
        duration: Optional[Seconds] = None,
        force_opus_sampling_rate: Optional[int] = None,
    ):
        assert isinstance(
            path_or_fd, (str, Path)
        ), f"Cannot use an ffmpeg subprocess to read from path of type: '{type(path_or_fd)}'"
        return read_opus_ffmpeg(
            path=path_or_fd,
            offset=offset,
            duration=duration,
            force_opus_sampling_rate=force_opus_sampling_rate,
        )

    def handles_special_case(self, path_or_fd: Union[Pathlike, FileObject]) -> bool:
        return isinstance(path_or_fd, (str, Path)) and str(path_or_fd).lower().endswith(
            ".opus"
        )

    def is_applicable(self, path_or_fd: Union[Pathlike, FileObject]) -> bool:
        return self.handles_special_case(path_or_fd)


class Sph2pipeSubprocessBackend(AudioBackend):
    def read_audio(
        self,
        path_or_fd: Union[Pathlike, FileObject],
        offset: Seconds = 0.0,
        duration: Optional[Seconds] = None,
        force_opus_sampling_rate: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        assert isinstance(
            path_or_fd, (str, Path)
        ), f"Cannot use an sph2pipe subprocess to read from path of type: '{type(path_or_fd)}'"
        return read_sph(
            sph_path=path_or_fd,
            offset=offset,
            duration=duration,
        )

    def handles_special_case(self, path_or_fd: Union[Pathlike, FileObject]) -> bool:
        return isinstance(path_or_fd, (str, Path)) and str(path_or_fd).lower().endswith(
            ".sph"
        )

    def is_applicable(self, path_or_fd: Union[Pathlike, FileObject]) -> bool:
        return self.handles_special_case(path_or_fd)


class FfmpegTorchaudioStreamerBackend(AudioBackend):
    def read_audio(
        self,
        path_or_fd: Union[Pathlike, FileObject],
        offset: Seconds = 0.0,
        duration: Optional[Seconds] = None,
        force_opus_sampling_rate: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        return torchaudio_ffmpeg_load(
            path_or_fileobj=path_or_fd,
            offset=offset,
            duration=duration,
        )

    def handles_special_case(self, path_or_fd: Union[Pathlike, FileObject]) -> bool:
        return torchaudio_supports_ffmpeg() and isinstance(path_or_fd, BytesIO)

    def is_applicable(self, path_or_fd: Union[Pathlike, FileObject]) -> bool:
        # Technically it's applicable with regular files as well, but for now
        # we're not enabling that feature.
        return torchaudio_supports_ffmpeg() and isinstance(path_or_fd, BytesIO)


class TorchaudioDefaultBackend(AudioBackend):
    def read_audio(
        self,
        path_or_fd: Union[Pathlike, FileObject],
        offset: Seconds = 0.0,
        duration: Optional[Seconds] = None,
        force_opus_sampling_rate: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        return torchaudio_load(
            path_or_fd=path_or_fd,
            offset=offset,
            duration=duration,
        )


class TorchaudioFFMPEGBackend(AudioBackend):
    """
    A new FFMPEG backend available in torchaudio 2.0.
    It should be free from many issues of soundfile and sox_io backends.
    """

    def read_audio(
        self,
        path_or_fd: Union[Pathlike, FileObject],
        offset: Seconds = 0.0,
        duration: Optional[Seconds] = None,
        force_opus_sampling_rate: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        return torchaudio_2_ffmpeg_load(
            path_or_fd=path_or_fd,
            offset=offset,
            duration=duration,
        )

    def is_applicable(self, path_or_fd: Union[Pathlike, FileObject]) -> bool:
        """
        FFMPEG backend requires at least Torchaudio 2.0.
        For version == 2.0.x, we also need env var TORCHAUDIO_USE_BACKEND_DISPATCHER=1
        For version >= 2.1.x, this will already be the default.
        """
        return torchaudio_2_0_ffmpeg_enabled()


class LibsndfileBackend(AudioBackend):
    """
    A backend that uses PySoundFile.

    .. note:: PySoundFile has issues on MacOS because of the way its CFFI bindings are implemented.
        For now, we disable it on this platform.
        See: https://github.com/bastibe/python-soundfile/issues/331
    """

    def read_audio(
        self,
        path_or_fd: Union[Pathlike, FileObject],
        offset: Seconds = 0.0,
        duration: Optional[Seconds] = None,
        force_opus_sampling_rate: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        return soundfile_load(
            path_or_fd=path_or_fd,
            offset=offset,
            duration=duration,
        )

    def handles_special_case(self, path_or_fd: Union[Pathlike, FileObject]) -> bool:
        return (
            not (sys.platform == "darwin")
            and isinstance(path_or_fd, BytesIO)
            and not torchaudio_2_0_ffmpeg_enabled()  # FFMPEG is preferable to this hack.
        )

    def is_applicable(self, path_or_fd: Union[Pathlike, FileObject]) -> bool:
        # Technically it's applicable with regular files as well, but for now
        # we're not enabling that feature.
        return not (sys.platform == "darwin") and isinstance(path_or_fd, BytesIO)


class AudioreadBackend(AudioBackend):
    def read_audio(
        self,
        path_or_fd: Union[Pathlike, FileObject],
        offset: Seconds = 0.0,
        duration: Optional[Seconds] = None,
        force_opus_sampling_rate: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        return audioread_load(
            path_or_file=path_or_fd,
            offset=offset,
            duration=duration,
        )


class CompositeAudioBackend(AudioBackend):
    def __init__(self, backends: List[AudioBackend]):
        self.backends = backends

    def read_audio(
        self,
        path_or_fd: Union[Pathlike, FileObject],
        offset: Seconds = 0.0,
        duration: Optional[Seconds] = None,
        force_opus_sampling_rate: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        candidates = []
        for b in self.backends:
            if b.handles_special_case(path_or_fd):
                candidates.append(b)

        assert len(candidates) < 2, (
            f"CompositeAudioBackend has more than one sub-backend that "
            f"handles a given special case for input '{path_or_fd}'"
        )

        if len(candidates) == 1:
            try:
                return candidates[0].read_audio(
                    path_or_fd=path_or_fd,
                    offset=offset,
                    duration=duration,
                    force_opus_sampling_rate=force_opus_sampling_rate,
                )
            except Exception as e:
                raise AudioLoadingError(
                    f"Reading audio from '{path_or_fd}' failed. Details: {type(e)}: {str(e)}"
                )

        exceptions = []
        for b in self.backends:
            if b.is_applicable(path_or_fd):
                try:
                    return b.read_audio(
                        path_or_fd=path_or_fd,
                        offset=offset,
                        duration=duration,
                        force_opus_sampling_rate=force_opus_sampling_rate,
                    )
                except Exception as e:
                    msg = f"Exception #{len(exceptions)} ({type(b)}): "
                    if verbose_audio_loading_exceptions():
                        exceptions.append(f"{msg}{traceback.format_exc()}")
                    else:
                        exceptions.append(f"{msg}{type(e)}: {str(e)}")

        if not exceptions:
            raise AudioLoadingError(
                f"No applicable backend found for input: '{path_or_fd}'"
            )
        else:
            NL = "\n"
            maybe_info = (
                ""
                if verbose_audio_loading_exceptions()
                else "\nSet LHOTSE_AUDIO_LOADING_EXCEPTION_VERBOSE=1 environment variable for full stack traces."
            )
            raise AudioLoadingError(
                f"Reading audio from '{path_or_fd}' failed. Details:{NL}{NL.join(exceptions)}{maybe_info}"
            )


def verbose_audio_loading_exceptions() -> bool:
    return os.environ.get("LHOTSE_AUDIO_LOADING_EXCEPTION_VERBOSE") == "1"


@lru_cache(maxsize=1)
def get_default_audio_backend():
    """
    Return a backend that can be used to read all audio formats supported by Lhotse.

    It first looks for special cases that need very specific handling
    (such as: opus, sphere/shorten, in-memory buffers)
    and tries to match them against relevant audio backends.

    Then, it tries to use several audio loading libraries (torchaudio, soundfile, audioread).
    In case the first fails, it tries the next one, and so on.
    """
    return CompositeAudioBackend(
        [
            # First handle special cases: OPUS and SPHERE (SPHERE may be encoded with shorten,
            #   which can only be decoded by binaries "shorten" and "sph2pipe").
            FfmpegSubprocessOpusBackend(),
            Sph2pipeSubprocessBackend(),
            # New FFMPEG backend available only in torchaudio 2.0.x+
            TorchaudioFFMPEGBackend(),
            # Prefer libsndfile for in-memory buffers only
            LibsndfileBackend(),
            # Torchaudio should be able to deal with most audio types...
            TorchaudioDefaultBackend(),
            # ... if not, try audioread...
            AudioreadBackend(),
            # ... oops.
        ]
    )


class LibsndfileCompatibleAudioInfo(NamedTuple):
    channels: int
    frames: int
    samplerate: int
    duration: float


def info(
    path: Pathlike,
    force_opus_sampling_rate: Optional[int] = None,
    force_read_audio: bool = False,
) -> LibsndfileCompatibleAudioInfo:

    if force_read_audio:
        # This is a reliable fallback for situations when the user knows that audio files do not
        # have duration metadata in their headers.
        # We will use "audioread" backend that spawns an ffmpeg process, reads the audio,
        # and computes the duration.
        return audioread_info(str(path))

    if path.suffix.lower() == ".opus":
        # We handle OPUS as a special case because we might need to force a certain sampling rate.
        return opus_info(path, force_opus_sampling_rate=force_opus_sampling_rate)

    elif path.suffix.lower() == ".sph":
        # We handle SPHERE as another special case because some old codecs (i.e. "shorten" codec)
        # can't be handled by neither pysoundfile nor pyaudioread.
        return sph_info(path)

    try:
        # Try to parse the file using torchaudio first.
        return torchaudio_info(path)
    except:
        try:
            # Try to parse the file using pysoundfile as a fallback.
            import soundfile as sf

            return sf.info(str(path))
        except:
            # Try to parse the file using audioread as the last fallback.
            return audioread_info(str(path))
            # If both fail, then Python 3 will display both exception messages.


@lru_cache(maxsize=1)
def torchaudio_supports_ffmpeg() -> bool:
    """
    Returns ``True`` when torchaudio version is at least 0.12.0, which
    has support for FFMPEG streamer API.
    """
    # If user has disabled ffmpeg-torchaudio, we don't need to check the version.
    if not FFMPEG_TORCHAUDIO_INFO_ENABLED:
        return False

    import torchaudio
    from packaging import version

    return version.parse(torchaudio.__version__) >= version.parse("0.12.0")


@lru_cache(maxsize=1)
def torchaudio_2_0_ffmpeg_enabled() -> bool:
    """
    Returns ``True`` when torchaudio.load supports "ffmpeg" backend.
    This requires either version 2.1.x+ or 2.0.x with env var TORCHAUDIO_USE_BACKEND_DISPATCHER=1.
    """
    import torchaudio
    from packaging import version

    ver = version.parse(torchaudio.__version__)
    if ver == version.parse("2.0.0"):
        return os.environ.get("TORCHAUDIO_USE_BACKEND_DISPATCHER", "0") == "1"
    if ver >= version.parse("2.1.0"):
        return True
    return False


@lru_cache(maxsize=1)
def torchaudio_soundfile_supports_format() -> bool:
    """
    Returns ``True`` when torchaudio version is at least 0.9.0, which
    has support for ``format`` keyword arg in ``torchaudio.save()``.
    """
    import torchaudio
    from packaging import version

    return version.parse(torchaudio.__version__) >= version.parse("0.9.0")


def torchaudio_info(
    path_or_fileobj: Union[Path, str, BytesIO]
) -> LibsndfileCompatibleAudioInfo:
    """
    Return an audio info data structure that's a compatible subset of ``pysoundfile.info()``
    that we need to create a ``Recording`` manifest.
    """
    import torchaudio

    if torchaudio_2_0_ffmpeg_enabled():
        # Torchaudio 2.0 with official "ffmpeg" backend should solve all the special cases below.
        info = torchaudio.info(path_or_fileobj, backend="ffmpeg")
        return LibsndfileCompatibleAudioInfo(
            channels=info.num_channels,
            frames=info.num_frames,
            samplerate=int(info.sample_rate),
            duration=info.num_frames / info.sample_rate,
        )

    is_mp3 = isinstance(path_or_fileobj, (str, Path)) and str(path_or_fileobj).endswith(
        ".mp3"
    )
    is_fileobj = isinstance(path_or_fileobj, BytesIO)
    if (is_mp3 or is_fileobj) and torchaudio_supports_ffmpeg():
        # Torchaudio 0.12 has a new StreamReader API that uses ffmpeg.
        #
        # They dropped support for using sox bindings in torchaudio.info
        # for MP3 files and implicitly delegate the call to ffmpeg.
        # Unfortunately, they always return num_frames/num_samples = 0,
        # as explained here: https://github.com/pytorch/audio/issues/2524
        # We have to work around by streaming the MP3 and counting the number
        # of samples.
        #
        # Unfortunately torchaudio also has issues with reading from file objects
        # sometimes, which apparently we can work around by using StreamReader API.
        # See:
        # - https://github.com/pytorch/audio/issues/2524#issuecomment-1223901818
        # - https://github.com/pytorch/audio/issues/2662
        from torchaudio.io import StreamReader

        streamer = StreamReader(src=str(path_or_fileobj) if is_mp3 else path_or_fileobj)
        assert streamer.num_src_streams == 1, (
            "Lhotse doesn't support files with more than one source stream yet "
            "(not to be confused with multi-channel)."
        )
        info = streamer.get_src_stream_info(streamer.default_audio_stream)
        streamer.add_basic_audio_stream(
            frames_per_chunk=int(info.sample_rate),
        )
        tot_samples = 0
        for (chunk,) in streamer.stream():
            tot_samples += chunk.shape[0]
        return LibsndfileCompatibleAudioInfo(
            channels=info.num_channels,
            frames=tot_samples,
            samplerate=int(info.sample_rate),
            duration=tot_samples / info.sample_rate,
        )

    info = torchaudio.info(path_or_fileobj)
    return LibsndfileCompatibleAudioInfo(
        channels=info.num_channels,
        frames=info.num_frames,
        samplerate=int(info.sample_rate),
        duration=info.num_frames / info.sample_rate,
    )


def torchaudio_load(
    path_or_fd: Pathlike, offset: Seconds = 0, duration: Optional[Seconds] = None
) -> Tuple[np.ndarray, int]:
    import torchaudio

    # Need to grab the "info" about sampling rate before reading to compute
    # the number of samples provided in offset / num_frames.
    frame_offset = 0
    num_frames = -1
    if offset > 0 or duration is not None:
        audio_info = torchaudio_info(path_or_fd)
        if offset > 0:
            frame_offset = compute_num_samples(offset, audio_info.samplerate)
        if duration is not None:
            num_frames = compute_num_samples(duration, audio_info.samplerate)
    if isinstance(path_or_fd, IOBase):
        # Set seek pointer to the beginning of the file as torchaudio.info
        # might have left it at the end of the header
        path_or_fd.seek(0)
    audio, sampling_rate = torchaudio.load(
        path_or_fd,
        frame_offset=frame_offset,
        num_frames=num_frames,
    )
    return audio.numpy(), int(sampling_rate)


def torchaudio_2_ffmpeg_load(
    path_or_fd: Pathlike, offset: Seconds = 0, duration: Optional[Seconds] = None
) -> Tuple[np.ndarray, int]:
    import torchaudio

    # Need to grab the "info" about sampling rate before reading to compute
    # the number of samples provided in offset / num_frames.
    frame_offset = 0
    num_frames = -1
    if offset > 0 or duration is not None:
        audio_info = torchaudio.info(path_or_fd, backend="ffmpeg")
        if offset > 0:
            frame_offset = compute_num_samples(offset, audio_info.samplerate)
        if duration is not None:
            num_frames = compute_num_samples(duration, audio_info.samplerate)
    if isinstance(path_or_fd, IOBase):
        # Set seek pointer to the beginning of the file as torchaudio.info
        # might have left it at the end of the header
        path_or_fd.seek(0)
    audio, sampling_rate = torchaudio.load(
        path_or_fd,
        frame_offset=frame_offset,
        num_frames=num_frames,
        backend="ffmpeg",
    )
    return audio.numpy(), int(sampling_rate)


def torchaudio_ffmpeg_load(
    path_or_fileobj: Union[Path, str, BytesIO],
    offset: Seconds = 0,
    duration: Optional[Seconds] = None,
) -> Tuple[np.ndarray, int]:
    import torchaudio

    if not torchaudio_supports_ffmpeg():
        raise RuntimeError(
            "Using FFMPEG streamer backend for reading is supported only "
            "with PyTorch 1.12+ and torchaudio 0.12+"
        )

    if isinstance(path_or_fileobj, Path):
        path_or_fileobj = str(path_or_fileobj)

    streamer = torchaudio.io.StreamReader(src=path_or_fileobj)
    assert streamer.num_src_streams == 1, (
        "Lhotse doesn't support files with more than one source stream yet "
        "(not to be confused with multi-channel)."
    )
    info = streamer.get_src_stream_info(streamer.default_audio_stream)
    sampling_rate = int(info.sample_rate)

    if duration is not None:
        # Try to read whole audio in a single chunk.
        streamer.add_basic_audio_stream(
            frames_per_chunk=compute_num_samples(duration, sampling_rate)
        )
        streamer.seek(offset)
        (audio,) = next(streamer.stream())
        audio = audio.transpose(0, 1)
    else:
        # Read in 1 second chunks and concatenate (we don't know how much audio is incoming)
        streamer.add_basic_audio_stream(frames_per_chunk=sampling_rate)
        streamer.seek(offset)
        audio = torch.cat([t.transpose(0, 1) for t, in streamer.stream()], dim=0)

    # Return shape (num_channels, num_samples)
    return audio.numpy(), sampling_rate


def soundfile_load(
    path_or_fd: Pathlike, offset: Seconds = 0, duration: Optional[Seconds] = None
) -> Tuple[np.ndarray, int]:
    import soundfile as sf

    with sf.SoundFile(path_or_fd) as sf_desc:
        sampling_rate = sf_desc.samplerate
        if offset > 0:
            # Seek to the start of the target read
            sf_desc.seek(compute_num_samples(offset, sampling_rate))
        if duration is not None:
            frame_duration = compute_num_samples(duration, sampling_rate)
        else:
            frame_duration = -1
        # Load the target number of frames, and transpose to match librosa form
        return (
            sf_desc.read(frames=frame_duration, dtype=np.float32, always_2d=False).T,
            int(sampling_rate),
        )


def audioread_info(path: Pathlike) -> LibsndfileCompatibleAudioInfo:
    """
    Return an audio info data structure that's a compatible subset of ``pysoundfile.info()``
    that we need to create a ``Recording`` manifest.
    """
    import audioread

    # We just read the file and compute the number of samples
    # -- no other method seems fully reliable...
    with audioread.audio_open(
        path, backends=_available_audioread_backends()
    ) as input_file:
        shape = audioread_load(input_file)[0].shape
        if len(shape) == 1:
            num_samples = shape[0]
        else:
            num_samples = shape[1]
        return LibsndfileCompatibleAudioInfo(
            channels=input_file.channels,
            frames=num_samples,
            samplerate=int(input_file.samplerate),
            duration=num_samples / input_file.samplerate,
        )


@lru_cache(maxsize=1)
def _available_audioread_backends():
    """
    Reduces the overhead of ``audioread.audio_open()`` when called repeatedly
    by caching the results of scanning for FFMPEG etc.
    """
    import audioread

    backends = audioread.available_backends()
    logging.info(f"Using audioread. Available backends: {backends}")
    return backends


def audioread_load(
    path_or_file: Union[Pathlike, FileObject],
    offset: Seconds = 0.0,
    duration: Seconds = None,
    dtype=np.float32,
):
    """Load an audio buffer using audioread.
    This loads one block at a time, and then concatenates the results.

    This function is based on librosa:
    https://github.com/librosa/librosa/blob/main/librosa/core/audio.py#L180
    """
    import audioread

    @contextmanager
    def file_handle():
        if isinstance(path_or_file, (str, Path)):
            yield audioread.audio_open(
                path_or_file, backends=_available_audioread_backends()
            )
        else:
            yield path_or_file

    y = []
    with file_handle() as input_file:
        sr_native = input_file.samplerate
        n_channels = input_file.channels

        s_start = int(np.round(sr_native * offset)) * n_channels

        if duration is None:
            s_end = np.inf
        else:
            s_end = s_start + (int(np.round(sr_native * duration)) * n_channels)

        n = 0

        for frame in input_file:
            frame = _buf_to_float(frame, dtype=dtype)
            n_prev = n
            n = n + len(frame)

            if n < s_start:
                # offset is after the current frame
                # keep reading
                continue

            if s_end < n_prev:
                # we're off the end.  stop reading
                break

            if s_end < n:
                # the end is in this frame.  crop.
                frame = frame[: s_end - n_prev]

            if n_prev <= s_start <= n:
                # beginning is in this frame
                frame = frame[(s_start - n_prev) :]

            # tack on the current frame
            y.append(frame)

    if y:
        y = np.concatenate(y)
        if n_channels > 1:
            y = y.reshape((-1, n_channels)).T
    else:
        y = np.empty(0, dtype=dtype)

    return y, int(sr_native)


def _buf_to_float(x, n_bytes=2, dtype=np.float32):
    """Convert an integer buffer to floating point values.
    This is primarily useful when loading integer-valued wav data
    into numpy arrays.

    This function is based on librosa:
    https://github.com/librosa/librosa/blob/main/librosa/util/utils.py#L1312

    Parameters
    ----------
    x : np.ndarray [dtype=int]
        The integer-valued data buffer
    n_bytes : int [1, 2, 4]
        The number of bytes per sample in ``x``
    dtype : numeric type
        The target output type (default: 32-bit float)
    Returns
    -------
    x_float : np.ndarray [dtype=float]
        The input data buffer cast to floating point
    """

    # Invert the scale of the data
    scale = 1.0 / float(1 << ((8 * n_bytes) - 1))

    # Construct the format string
    fmt = "<i{:d}".format(n_bytes)

    # Rescale and format the data buffer
    return scale * np.frombuffer(x, fmt).astype(dtype)


def assert_and_maybe_fix_num_samples(
    audio: np.ndarray,
    offset: Seconds,
    duration: Optional[Seconds],
    recording: Recording,
) -> np.ndarray:
    # When resampling in high sampling rates (48k -> 44.1k)
    # it is difficult to estimate how sox will perform rounding;
    # we will just add/remove one sample to be consistent with
    # what we have estimated.
    # This effect is exacerbated by chaining multiple augmentations together.
    expected_num_samples = compute_num_samples(
        duration=duration if duration is not None else recording.duration - offset,
        sampling_rate=recording.sampling_rate,
    )
    diff = expected_num_samples - audio.shape[1]
    if diff == 0:
        return audio  # this is normal condition
    allowed_diff = int(
        ceil(LHOTSE_AUDIO_DURATION_MISMATCH_TOLERANCE * recording.sampling_rate)
    )
    if 0 < diff <= allowed_diff:
        audio = np.pad(audio, ((0, 0), (0, diff)), mode="reflect")
        return audio
    elif -allowed_diff <= diff < 0:
        audio = audio[:, :diff]
        return audio
    else:
        raise ValueError(
            "The number of declared samples in the recording diverged from the one obtained "
            f"when loading audio (offset={offset}, duration={duration}). "
            f"This could be internal Lhotse's error or a faulty transform implementation. "
            "Please report this issue in Lhotse and show the "
            f"following: diff={diff}, audio.shape={audio.shape}, recording={recording}"
        )


def opus_info(
    path: Pathlike, force_opus_sampling_rate: Optional[int] = None
) -> LibsndfileCompatibleAudioInfo:
    samples, sampling_rate = read_opus(
        path, force_opus_sampling_rate=force_opus_sampling_rate
    )
    return LibsndfileCompatibleAudioInfo(
        channels=samples.shape[0],
        frames=samples.shape[1],
        samplerate=int(sampling_rate),
        duration=samples.shape[1] / sampling_rate,
    )


def read_opus(
    path: Pathlike,
    offset: Seconds = 0.0,
    duration: Optional[Seconds] = None,
    force_opus_sampling_rate: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """
    Reads OPUS files either using torchaudio or ffmpeg.
    Torchaudio is faster, but if unavailable for some reason,
    we fallback to a slower ffmpeg-based implementation.

    :return: a tuple of audio samples and the sampling rate.
    """
    # TODO: Revisit using torchaudio backend for OPUS
    #       once it's more thoroughly benchmarked against ffmpeg
    #       and has a competitive I/O speed.
    #       See: https://github.com/pytorch/audio/issues/1994
    # try:
    #     return read_opus_torchaudio(
    #         path=path,
    #         offset=offset,
    #         duration=duration,
    #         force_opus_sampling_rate=force_opus_sampling_rate,
    #     )
    # except:
    return read_opus_ffmpeg(
        path=path,
        offset=offset,
        duration=duration,
        force_opus_sampling_rate=force_opus_sampling_rate,
    )


def read_opus_torchaudio(
    path: Pathlike,
    offset: Seconds = 0.0,
    duration: Optional[Seconds] = None,
    force_opus_sampling_rate: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """
    Reads OPUS files using torchaudio.
    This is just running ``tochaudio.load()``, but we take care of extra resampling if needed.

    :return: a tuple of audio samples and the sampling rate.
    """
    audio, sampling_rate = torchaudio_load(
        path_or_fd=path, offset=offset, duration=duration
    )

    if force_opus_sampling_rate is None or force_opus_sampling_rate == sampling_rate:
        return audio, sampling_rate

    resampler = Resample(
        source_sampling_rate=sampling_rate,
        target_sampling_rate=force_opus_sampling_rate,
    )
    resampled_audio = resampler(audio)
    return resampled_audio, force_opus_sampling_rate


def read_opus_ffmpeg(
    path: Pathlike,
    offset: Seconds = 0.0,
    duration: Optional[Seconds] = None,
    force_opus_sampling_rate: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """
    Reads OPUS files using ffmpeg in a shell subprocess.
    Unlike audioread, correctly supports offsets and durations for reading short chunks.
    Optionally, we can force ffmpeg to resample to the true sampling rate (if we know it up-front).

    :return: a tuple of audio samples and the sampling rate.
    """
    # Construct the ffmpeg command depending on the arguments passed.
    cmd = "ffmpeg -threads 1"
    sampling_rate = 48000
    # Note: we have to add offset and duration options (-ss and -t) BEFORE specifying the input
    #       (-i), otherwise ffmpeg will decode everything and trim afterwards...
    if offset > 0:
        cmd += f" -ss {offset}"
    if duration is not None:
        cmd += f" -t {duration}"
    # Add the input specifier after offset and duration.
    cmd += f" -i {path}"
    # Optionally resample the output.
    if force_opus_sampling_rate is not None:
        cmd += f" -ar {force_opus_sampling_rate}"
        sampling_rate = force_opus_sampling_rate
    # Read audio samples directly as float32.
    cmd += " -f f32le -threads 1 pipe:1"
    # Actual audio reading.
    proc = run(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    raw_audio = proc.stdout
    audio = np.frombuffer(raw_audio, dtype=np.float32)
    # Determine if the recording is mono or stereo and decode accordingly.
    try:
        channel_string = parse_channel_from_ffmpeg_output(proc.stderr)
        if channel_string == "stereo":
            new_audio = np.empty((2, audio.shape[0] // 2), dtype=np.float32)
            new_audio[0, :] = audio[::2]
            new_audio[1, :] = audio[1::2]
            audio = new_audio
        elif channel_string == "mono":
            audio = audio.reshape(1, -1)
        else:
            raise NotImplementedError(
                f"Unknown channel description from ffmpeg: {channel_string}"
            )
    except ValueError as e:
        raise AudioLoadingError(
            f"{e}\nThe ffmpeg command for which the program failed is: '{cmd}', error code: {proc.returncode}"
        )
    return audio, sampling_rate


def parse_channel_from_ffmpeg_output(ffmpeg_stderr: bytes) -> str:
    # ffmpeg will output line such as the following, amongst others:
    # "Stream #0:0: Audio: pcm_f32le, 16000 Hz, mono, flt, 512 kb/s"
    # but sometimes it can be "Stream #0:0(eng):", which we handle with regexp
    pattern = re.compile(r"^\s*Stream #0:0.*: Audio: pcm_f32le.+(mono|stereo).+\s*$")
    for line in ffmpeg_stderr.splitlines():
        try:
            line = line.decode()
        except UnicodeDecodeError:
            # Why can we get UnicodeDecoderError from ffmpeg output?
            # Because some files may contain the metadata, including a short description of the recording,
            # which may be encoded in arbitrarily encoding different than ASCII/UTF-8, such as latin-1,
            # and Python will not automatically recognize that.
            # We simply ignore these lines as they won't have any relevant information for us.
            continue
        match = pattern.match(line)
        if match is not None:
            return match.group(1)
    raise ValueError(
        f"Could not determine the number of channels for OPUS file from the following ffmpeg output "
        f"(shown as bytestring due to avoid possible encoding issues):\n{str(ffmpeg_stderr)}"
    )


def sph_info(path: Pathlike) -> LibsndfileCompatibleAudioInfo:
    samples, sampling_rate = read_sph(path)
    return LibsndfileCompatibleAudioInfo(
        channels=samples.shape[0],
        frames=samples.shape[1],
        samplerate=int(sampling_rate),
        duration=samples.shape[1] / sampling_rate,
    )


def read_sph(
    sph_path: Pathlike, offset: Seconds = 0.0, duration: Optional[Seconds] = None
) -> Tuple[np.ndarray, int]:
    """
    Reads SPH files using sph2pipe in a shell subprocess.
    Unlike audioread, correctly supports offsets and durations for reading short chunks.

    :return: a tuple of audio samples and the sampling rate.
    """

    sph_path = Path(sph_path)

    # Construct the sph2pipe command depending on the arguments passed.
    cmd = f"sph2pipe -f wav -p -t {offset}:"

    if duration is not None:
        cmd += f"{round(offset + duration, 5)}"
    # Add the input specifier after offset and duration.
    cmd += f" {sph_path}"

    # Actual audio reading.
    try:
        proc = BytesIO(
            run(cmd, shell=True, check=True, stdout=PIPE, stderr=PIPE).stdout
        )
    except CalledProcessError as e:
        if e.returncode == 127:
            raise ValueError(
                "It seems that 'sph2pipe' binary is not installed; "
                "did you run 'lhotse install-sph2pipe'?"
            )
        else:
            raise

    import soundfile as sf

    with sf.SoundFile(proc) as sf_desc:
        audio, sampling_rate = sf_desc.read(dtype=np.float32), sf_desc.samplerate
        audio = audio.reshape(1, -1) if sf_desc.channels == 1 else audio.T

    return audio, sampling_rate


def torchaudio_save_flac_safe(
    dest: Union[str, Path, BytesIO],
    src: Union[torch.Tensor, np.ndarray],
    sample_rate: int,
    *args,
    **kwargs,
):
    import torchaudio

    src = torch.as_tensor(src)
    saving_flac = kwargs.get("format") == "flac" or (
        not isinstance(dest, BytesIO) and str(dest).endswith(".flac")
    )
    if torchaudio_soundfile_supports_format() and saving_flac:
        # Prefer saving with soundfile backend whenever possible to avoid issue:
        # https://github.com/pytorch/audio/issues/2662
        # Saving with sox_io backend to FLAC may corrupt the file.
        torchaudio.backend.soundfile_backend.save(
            dest,
            src,
            sample_rate=sample_rate,
            format=kwargs.pop("format", "flac"),
            bits_per_sample=kwargs.pop("bits_per_sample", 16),
            *args,
            **kwargs,
        )
    else:
        torchaudio.backend.sox_io_backend.save(
            dest, src, sample_rate=sample_rate, *args, **kwargs
        )


class AudioLoadingError(Exception):
    pass


class DurationMismatchError(Exception):
    pass


@contextmanager
def suppress_audio_loading_errors(enabled: bool = True):
    """
    Context manager that suppresses errors related to audio loading.
    Emits warning to the console.
    """
    with suppress_and_warn(
        AudioLoadingError,
        DurationMismatchError,
        NonPositiveEnergyError,
        enabled=enabled,
    ):
        yield


def null_result_on_audio_loading_error(func: Callable) -> Callable:
    """
    This is a decorator that makes a function return None when reading audio with Lhotse failed.

    Example::

        >>> @null_result_on_audio_loading_error
        ... def func_loading_audio(rec):
        ...     audio = rec.load_audio()  # if this fails, will return None instead
        ...     return other_func(audio)

    Another example::

        >>> # crashes on loading audio
        >>> audio = load_audio(cut)
        >>> # does not crash on loading audio, return None instead
        >>> maybe_audio: Optional = null_result_on_audio_loading_error(load_audio)(cut)
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Optional:
        with suppress_audio_loading_errors():
            return func(*args, **kwargs)

    return wrapper
