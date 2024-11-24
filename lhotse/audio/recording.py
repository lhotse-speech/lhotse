from dataclasses import dataclass
from io import BytesIO
from math import ceil, isclose
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from _decimal import ROUND_HALF_UP

from lhotse.audio.backend import get_current_audio_backend, info, save_audio
from lhotse.audio.source import AudioSource
from lhotse.audio.utils import (
    AudioLoadingError,
    DurationMismatchError,
    VideoInfo,
    get_audio_duration_mismatch_tolerance,
)
from lhotse.augmentation import (
    AudioTransform,
    DereverbWPE,
    LoudnessNormalization,
    Narrowband,
    Resample,
    ReverbWithImpulseResponse,
    Speed,
    Tempo,
    Volume,
)
from lhotse.utils import (
    Pathlike,
    Seconds,
    SetContainingAnything,
    asdict_nonull,
    compute_num_samples,
    fastcopy,
    ifnone,
    perturb_num_samples,
    rich_exception_info,
)

Channels = Union[int, List[int]]


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

    Internally, Lhotse supports multiple audio backends to read audio file.
    By default, we try to use libsoundfile, then torchaudio (with FFMPEG integration starting with torchaudio 2.1),
    and then audioread (which is an ffmpeg CLI wrapper).
    For sphere files we prefer to use sph2pipe binary as it can work with certain unique encodings such as "shorten".

    Audio backends in Lhotse are configurable. See:

    * :func:`~lhotse.audio.backend.available_audio_backends`
    * :func:`~lhotse.audio.backend.audio_backend`,
    * :func:`~lhotse.audio.backend.get_current_audio_backend`
    * :func:`~lhotse.audio.backend.set_current_audio_backend`
    * :func:`~lhotse.audio.backend.get_default_audio_backend`


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

        See also: :class:`~lhotse.audio.recording.Recording`, :class:`~lhotse.cut.Cut`, :class:`~lhotse.cut.CutSet`.
    """

    id: str
    sources: List[AudioSource]
    sampling_rate: int
    num_samples: int
    duration: Seconds
    channel_ids: Optional[List[int]] = None
    transforms: Optional[List[Union[AudioTransform, Dict]]] = None

    def __post_init__(self):
        if self.channel_ids is None:
            self.channel_ids = sorted(
                cid for source in self.sources for cid in source.channels
            )
        assert (
            sum(source.has_video for source in self.sources) < 2
        ), "Lhotse does not currently support recordings with more than a single video stream."

    @property
    def video(self) -> Optional[VideoInfo]:
        s = self._video_source
        if s is None:
            return None
        return s.video

    @property
    def has_video(self) -> bool:
        return self._video_source is not None

    @property
    def _video_source(self) -> Optional[AudioSource]:
        for s in self.sources:
            if s.has_video:
                return s
        return None

    @property
    def is_in_memory(self) -> bool:
        return any(s.type == "memory" for s in self.sources)

    @property
    def is_placeholder(self) -> bool:
        return any(s.type == "shar" for s in self.sources)

    @property
    def num_channels(self) -> int:
        return len(self.channel_ids)

    @property
    def source_format(self) -> str:
        """Infer format of the audio sources.
        If all sources have the same format, return it.
        If sources have different formats, raise an error.
        """
        source_formats = list(set([s.format for s in self.sources]))

        if len(source_formats) == 1:
            # if all sources have the same format, return it
            return source_formats[0]
        else:
            # at the moment, we don't resolve different formats
            raise NotImplementedError(
                "Sources have different formats. Resolving to a single format not implemented."
            )

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
        if audio_info.video is not None:
            duration = audio_info.video.duration
            num_samples = compute_num_samples(duration, audio_info.samplerate)
        else:
            duration = audio_info.duration
            num_samples = audio_info.frames
        return Recording(
            id=recording_id,
            sampling_rate=audio_info.samplerate,
            num_samples=num_samples,
            duration=duration,
            sources=[
                AudioSource(
                    type="file",
                    channels=list(range(audio_info.channels)),
                    source=(
                        "/".join(path.parts[-relative_path_depth:])
                        if relative_path_depth is not None and relative_path_depth > 0
                        else str(path)
                    ),
                    video=audio_info.video,
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
        audio_info = get_current_audio_backend().info(stream)
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

        def _aslist(x):
            if isinstance(x, int):
                return [x]
            return x

        # Case #1: no opts specified, read audio without decoding and move it in memory.
        if all(opt is None for opt in (channels, offset, duration)) or (
            (channels is None or _aslist(channels) == self.channel_ids)
            and (offset is None or isclose(offset, 0.0))
            and (duration is None or isclose(duration, self.duration))
        ):
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
        audio = self.load_audio(
            channels=channels, offset=ifnone(offset, 0), duration=duration
        )
        stream = BytesIO()
        save_audio(stream, torch.from_numpy(audio), self.sampling_rate, format=format)
        channels = ifnone(channels, self.channel_ids)
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
        d = asdict_nonull(self)
        if self.transforms is not None:
            d["transforms"] = [
                t if isinstance(t, dict) else t.to_dict() for t in self.transforms
            ]
        return d

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

        # Micro-optimization for a number of audio loading cases:
        # if duration is very close to full recording,
        # just read everything, and we'll discard some samples at the end.
        orig_duration = duration
        if duration is not None and isclose(duration, self.duration, abs_tol=1e-3):
            duration = None

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
            tnfm if isinstance(tnfm, AudioTransform) else AudioTransform.from_dict(tnfm)
            for tnfm in self.transforms or []
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

        if self.has_video:
            # It's possible the audio and video durations are quite mismatched.
            # We'll pad audio with zeroes or truncate audio to accomodate the video,
            # when it's available
            audio = assert_and_maybe_fix_num_samples(
                audio,
                offset=offset,
                duration=orig_duration,
                recording=self,
                tolerance=1e6,
                pad_mode="constant",
            )
        else:
            # Transformation chains can introduce small mismatches in the number of samples:
            # we'll fix them here, or raise an error if they exceeded a tolerance threshold.
            audio = assert_and_maybe_fix_num_samples(
                audio, offset=offset, duration=orig_duration, recording=self
            )

        return audio

    @rich_exception_info
    def load_video(
        self,
        channels: Optional[Channels] = None,
        offset: Seconds = 0.0,
        duration: Optional[Seconds] = None,
        with_audio: bool = True,
        force_consistent_duration: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Read the video frames and audio samples from the underlying source (path, URL, unix pipe/command).

        :param channels: int or iterable of ints, a subset of channel IDs to read (reads all by default).
        :param offset: seconds, where to start reading the video (at offset 0 by default).
            Note that it is only efficient for local filesystem files, i.e. URLs and commands will read
            all the samples first and discard the unneeded ones afterwards.
        :param duration: seconds, indicates the total video time to read (starting from ``offset``).
        :param with_audio: bool, whether to load and return audio alongside video. True by default.
        :param force_consistent_duration: bool, if audio duration is different than video duration
            (as counted by ``num_frames / fps``), we'll either truncate or pad the audio with zeros.
            True by default.
        :return: a tuple of video tensor and optional audio tensor (or None).
        """

        assert self.has_video, f"Recording {self.id} has no video to load."

        assert offset <= self.duration, (
            f"Cannot load audio because the Recording's duration {self.duration}s "
            f"is smaller than the requested offset {offset}s."
        )

        for t in ifnone(self.transforms, ()):
            if isinstance(t, dict):
                assert t["name"] not in (
                    "Speed",
                    "Tempo",
                ), "Recording.load_video() does not support speed/tempo perturbation."
            else:
                assert not isinstance(
                    t, (Speed, Tempo)
                ), "Recording.load_video() does not support speed/tempo perturbation."

        if not with_audio:
            video, _ = self._video_source.load_video(
                offset=offset, duration=duration, with_audio=False
            )
            return video, None

        # Micro-optimization for a number of audio loading cases:
        # if duration is very close to full recording,
        # just read everything, and we'll discard some samples at the end.
        orig_duration = duration
        if duration is not None and isclose(duration, self.duration, abs_tol=1e-3):
            duration = None

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
            tnfm if isinstance(tnfm, AudioTransform) else AudioTransform.from_dict(tnfm)
            for tnfm in self.transforms or []
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
        video = None
        for source in self.sources:
            if source.has_video:
                video, samples = source.load_video(
                    offset=offset_aug,
                    duration=duration_aug,
                )
            else:
                samples = source.load_audio(offset=offset_aug, duration=duration_aug)

            # Case: source not requested (for audio, but it might be the only one with video)
            if not channels.intersection(source.channels):
                continue

            # Case: two-channel audio file but only one channel requested
            #       it might not be optimal to load all channels, but IDK if there's anything we can do about it
            channels_to_remove = [
                idx for idx, cid in enumerate(source.channels) if cid not in channels
            ]
            if channels_to_remove:
                samples = np.delete(samples, channels_to_remove, axis=0)
            samples_per_source.append(samples)

        assert video is not None

        # Stack all the samples from all the sources into a single array.
        audio = self._stack_audio_channels(samples_per_source)

        # We'll apply the transforms now (if any).
        for tfn in transforms:
            audio = tfn(audio, self.sampling_rate)

        if force_consistent_duration:
            # We want to keep audio and video duration identical by truncating/padding audio.
            audio = assert_and_maybe_fix_num_samples(
                audio,
                offset=offset,
                duration=video.shape[0] / self.video.fps,
                recording=self,
                # hack: "infinite" tolerance disables exceptions, i.e. 1min video and 1h audio => 1min audio
                tolerance=1e6,
                pad_mode="zero",
            )
        else:
            # Transformation chains can introduce small mismatches in the number of samples:
            # we'll fix them here, or raise an error if they exceeded a tolerance threshold.
            audio = assert_and_maybe_fix_num_samples(
                audio,
                offset=offset,
                duration=orig_duration,
                recording=self,
                pad_mode="reflect",
            )

        return video, torch.from_numpy(audio)

    def play_video(self):
        if self.has_video:
            from IPython.display import Video

            return Video(filename=self._video_source.source)

    def _stack_audio_channels(self, samples_per_source: List[np.ndarray]) -> np.ndarray:
        # There may be a mismatch in the number of samples between different channels. We
        # check if the mismatch is within a reasonable tolerance and if so, we pad
        # all channels to the length of the longest one.
        allowed_diff = int(
            compute_num_samples(
                get_audio_duration_mismatch_tolerance(),
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
                        f"greater than the allowed tolerance {get_audio_duration_mismatch_tolerance()}."
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

    def with_video_resolution(self, width: int, height: int) -> "Recording":
        return fastcopy(
            self,
            sources=[
                s.with_video_resolution(width=width, height=height)
                for s in self.sources
            ],
        )

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
        transforms.append(Speed(factor=factor))
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
        transforms.append(Tempo(factor=factor))
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
        transforms.append(Volume(factor=factor))
        return fastcopy(
            self,
            id=f"{self.id}_vp{factor}" if affix_id else self.id,
            transforms=transforms,
        )

    def narrowband(
        self, codec: str, restore_orig_sr: bool = True, affix_id: bool = True
    ) -> "Recording":
        """
        Return a new ``Recording`` that will lazily apply narrowband effect while loading audio.
            by affixing it with "_nb_{codec}".

        :return: a modified copy of the current ``Recording``.
        """
        transforms = self.transforms.copy() if self.transforms is not None else []
        transforms.append(
            Narrowband(
                codec=codec,
                source_sampling_rate=self.sampling_rate,
                restore_orig_sr=restore_orig_sr,
            ).to_dict()
        )
        new_num_samples = compute_num_samples(
            self.duration,
            self.sampling_rate if restore_orig_sr else 8000,
            rounding=ROUND_HALF_UP,
        )
        return fastcopy(
            self,
            id=f"{self.id}_nb_{codec}" if affix_id else self.id,
            num_samples=new_num_samples,
            sampling_rate=self.sampling_rate if restore_orig_sr else 8000,
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
        transforms.append(LoudnessNormalization(target=target))
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
        transforms.append(DereverbWPE())
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
        rir_channels: Optional[Sequence[int]] = None,
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
            )
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
        transforms.append(
            Resample(
                source_sampling_rate=self.sampling_rate,
                target_sampling_rate=sampling_rate,
            )
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
        try:
            transforms = data.pop("transforms")
            transforms = [AudioTransform.from_dict(t) for t in transforms]
        except KeyError:
            transforms = None
        return Recording(
            sources=[AudioSource.from_dict(s) for s in raw_sources],
            transforms=transforms,
            **data,
        )


def assert_and_maybe_fix_num_samples(
    audio: np.ndarray,
    offset: Seconds,
    duration: Optional[Seconds],
    recording: Recording,
    tolerance: Optional[Seconds] = None,
    pad_mode: str = "reflect",
) -> np.ndarray:
    # When resampling in high sampling rates (48k -> 44.1k)
    # it is difficult to estimate how sox will perform rounding;
    # we will just add/remove one sample to be consistent with
    # what we have estimated.
    # This effect is exacerbated by chaining multiple augmentations together.
    if tolerance is None:  # use Lhotse's default
        tolerance = get_audio_duration_mismatch_tolerance()
    expected_num_samples = compute_num_samples(
        duration=duration if duration is not None else recording.duration - offset,
        sampling_rate=recording.sampling_rate,
    )
    diff = expected_num_samples - audio.shape[1]
    if diff == 0:
        return audio  # this is normal condition
    allowed_diff = int(ceil(tolerance * recording.sampling_rate))
    if 0 < diff <= allowed_diff:
        audio = np.pad(audio, ((0, 0), (0, diff)), mode=pad_mode)
        return audio
    elif -allowed_diff <= diff < 0:
        audio = audio[:, :diff]
        return audio
    else:
        raise AudioLoadingError(
            "The number of declared samples in the recording diverged from the one obtained "
            f"when loading audio (offset={offset}, duration={duration}). "
            f"This could be internal Lhotse's error or a faulty transform implementation. "
            "Please report this issue in Lhotse and show the "
            f"following: diff={diff}, audio.shape={audio.shape}, recording={recording}"
        )
