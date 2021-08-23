import logging
import random
import re
import warnings
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import ROUND_HALF_UP
from functools import lru_cache, partial
from io import BytesIO
from itertools import islice
from math import ceil, sqrt
from pathlib import Path
from subprocess import PIPE, run
from typing import Any, Callable, Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
from tqdm.auto import tqdm

from lhotse.augmentation import AudioTransform, Resample, Speed, Tempo, Volume
from lhotse.serialization import Serializable
from lhotse.utils import (Decibels, NonPositiveEnergyError, Pathlike, Seconds, SetContainingAnything, SmartOpen,
                          asdict_nonull, compute_num_samples, exactly_one_not_null, fastcopy, ifnone,
                          index_by_id_and_check, perturb_num_samples, split_sequence)

Channels = Union[int, List[int]]


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
    """
    type: str
    channels: List[int]
    source: str

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
        and are of dtype `np.floatt32`.

        :param force_opus_sampling_rate: This parameter is only used when we detect an OPUS file.
            It will tell ffmpeg to resample OPUS to this sampling rate.
        """
        assert self.type in ('file', 'command', 'url')

        # TODO: refactor when another source type is added
        source = self.source

        if self.type == 'command':
            if offset != 0.0 or duration is not None:
                # TODO(pzelasko): How should we support chunking for commands?
                #                 We risk being very inefficient when reading many chunks from the same file
                #                 without some caching scheme, because we'll be re-running commands.
                warnings.warn('You requested a subset of a recording that is read from disk via a bash command. '
                              'Expect large I/O overhead if you are going to read many chunks like these, '
                              'since every time we will read the whole file rather than its subset.')
            source = BytesIO(run(self.source, shell=True, stdout=PIPE).stdout)
            samples, sampling_rate = read_audio(source, offset=offset, duration=duration)

        elif self.type == 'url':
            if offset != 0.0 or duration is not None:
                # TODO(pzelasko): How should we support chunking for URLs?
                #                 We risk being very inefficient when reading many chunks from the same file
                #                 without some caching scheme, because we'll be re-running commands.
                warnings.warn('You requested a subset of a recording that is read from URL. '
                              'Expect large I/O overhead if you are going to read many chunks like these, '
                              'since every time we will download the whole file rather than its subset.')
            with SmartOpen.open(self.source, 'rb') as f:
                source = BytesIO(f.read())
                samples, sampling_rate = read_audio(source, offset=offset, duration=duration)

        else:  # self.type == 'file'
            samples, sampling_rate = read_audio(
                source,
                offset=offset,
                duration=duration,
                force_opus_sampling_rate=force_opus_sampling_rate,
            )

        # explicit sanity check for duration as soundfile does not complain here
        if duration is not None:
            num_samples = samples.shape[0] if len(samples.shape) == 1 else samples.shape[1]
            available_duration = num_samples / sampling_rate
            if available_duration < duration - 1e-3:  # set the allowance as 1ms to avoid float error
                raise ValueError(
                    f'Requested more audio ({duration}s) than available ({available_duration}s)'
                )

        return samples.astype(np.float32)

    def with_path_prefix(self, path: Pathlike) -> 'AudioSource':
        if self.type != 'file':
            return self
        return fastcopy(self, source=str(Path(path) / self.source))

    def to_dict(self) -> dict:
        return asdict_nonull(self)

    @staticmethod
    def from_dict(data) -> 'AudioSource':
        return AudioSource(**data)


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
    transforms: Optional[List[Dict]] = None

    @staticmethod
    def from_file(
            path: Pathlike,
            recording_id: Optional[str] = None,
            relative_path_depth: Optional[int] = None,
            force_opus_sampling_rate: Optional[int] = None,
    ) -> 'Recording':
        """
        Read an audio file's header and create the corresponding ``Recording``.
        Suitable to use when each physical file represents a separate recording session.

        .. caution::
            If a recording session consists of multiple files (e.g. one per channel),
            it is advisable to create the ``Recording`` object manually, with each
            file represented as a separate ``AudioSource`` object.

        :param path: Path to an audio file supported by libsoundfile (pysoundfile).
        :param recording_id: recording id, when not specified ream the filename's stem ("x.wav" -> "x").
        :param relative_path_depth: optional int specifying how many last parts of the file path
            should be retained in the ``AudioSource``. By default writes the path as is.
        :param force_opus_sampling_rate: when specified, this value will be used as the sampling rate
            instead of the one we read from the manifest. This is useful for OPUS files that always
            have 48kHz rate and need to be resampled to the real one -- we will perform that operation
            "under-the-hood". For non-OPUS files this input is undefined.
        :return: a new ``Recording`` instance pointing to the audio file.
        """
        path = Path(path)
        if path.suffix.lower() == '.opus':
            # We handle OPUS as a special case because we might need to force a certain sampling rate.
            info = opus_info(path, force_opus_sampling_rate=force_opus_sampling_rate)
        elif path.suffix.lower() == '.sph':
            # We handle SPHERE as another special case because some old codecs (i.e. "shorten" codec)
            # can't be handled by neither pysoundfile nor pyaudioread.
            info = sph_info(path)
        else:
            try:
                # Try to parse the file using pysoundfile first.
                import soundfile as sf
                info = sf.info(str(path))
            except:
                # Try to parse the file using audioread as a fallback.
                info = audioread_info(str(path))
                # If both fail, then Python 3 will display both exception messages.
        return Recording(
            id=recording_id if recording_id is not None else path.stem,
            sampling_rate=info.samplerate,
            num_samples=info.frames,
            duration=info.duration,
            sources=[
                AudioSource(
                    type='file',
                    channels=list(range(info.channels)),
                    source=(
                        '/'.join(path.parts[-relative_path_depth:])
                        if relative_path_depth is not None and relative_path_depth > 0
                        else str(path)
                    )
                )
            ]
        )

    def to_dict(self) -> dict:
        return asdict_nonull(self)

    @property
    def num_channels(self):
        return sum(len(source.channels) for source in self.sources)

    @property
    def channel_ids(self):
        return sorted(cid for source in self.sources for cid in source.channels)

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
        if channels is None:
            channels = SetContainingAnything()
        else:
            channels = frozenset([channels] if isinstance(channels, int) else channels)
            recording_channels = frozenset(self.channel_ids)
            assert channels.issubset(recording_channels), "Requested to load audio from a channel " \
                                                          "that does not exist in the recording: " \
                                                          f"(recording channels: {recording_channels} -- " \
                                                          f"requested channels: {channels})"

        transforms = [AudioTransform.from_dict(params) for params in self.transforms or []]

        # Do a "backward pass" over data augmentation transforms to get the
        # offset and duration for loading a piece of the original audio.
        offset_aug, duration_aug = offset, duration
        for tfn in reversed(transforms):
            offset_aug, duration_aug = tfn.reverse_timestamps(
                offset=offset_aug,
                duration=duration_aug,
                sampling_rate=self.sampling_rate
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
                idx for idx, cid in enumerate(source.channels)
                if cid not in channels
            ]
            if channels_to_remove:
                samples = np.delete(samples, channels_to_remove, axis=0)
            samples_per_source.append(samples)

        # shape: (n_channels, n_samples)
        audio = np.vstack(samples_per_source)

        # We'll apply the transforms now (if any).
        for tfn in transforms:
            audio = tfn(audio, self.sampling_rate)

        # Transformation chains can introduce small mismatches in the number of samples:
        # we'll fix them here, or raise an error if they exceeded a tolerance threshold.
        audio = assert_and_maybe_fix_num_samples(
            audio,
            offset=offset,
            duration=duration,
            recording=self
        )

        return audio

    def _expected_num_samples(self, offset: Seconds, duration: Optional[Seconds]) -> int:
        if offset == 0 and duration is None:
            return self.num_samples
        duration = duration if duration is not None else self.duration - offset
        return compute_num_samples(duration, sampling_rate=self.sampling_rate)

    def with_path_prefix(self, path: Pathlike) -> 'Recording':
        return fastcopy(self, sources=[s.with_path_prefix(path) for s in self.sources])

    def perturb_speed(self, factor: float, affix_id: bool = True) -> 'Recording':
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
            id=f'{self.id}_sp{factor}' if affix_id else self.id,
            num_samples=new_num_samples,
            duration=new_duration,
            transforms=transforms
        )

    def perturb_tempo(self, factor: float, affix_id: bool = True) -> 'Recording':
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
            id=f'{self.id}_tp{factor}' if affix_id else self.id,
            num_samples=new_num_samples,
            duration=new_duration,
            transforms=transforms
        )

    def perturb_volume(self, factor: float, affix_id: bool = True) -> 'Recording':
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
            id=f'{self.id}_vp{factor}' if affix_id else self.id,
            transforms=transforms
        )

    def resample(self, sampling_rate: int) -> 'Recording':
        """
        Return a new ``Recording`` that will be lazily resampled while loading audio.
        :param sampling_rate: The new sampling rate.
        :return: A resampled ``Recording``.
        """
        transforms = self.transforms.copy() if self.transforms is not None else []
        transforms.append(
            Resample(source_sampling_rate=self.sampling_rate, target_sampling_rate=sampling_rate).to_dict()
        )
        new_num_samples = compute_num_samples(self.duration, sampling_rate, rounding=ROUND_HALF_UP)
        # Duration might need an adjustment when doing a non-trivial resampling
        # (e.g. 16000 -> 22050), where the resulting number of samples cannot
        # correspond to old duration exactly.
        new_duration = new_num_samples / sampling_rate
        return fastcopy(
            self,
            duration=new_duration,
            num_samples=new_num_samples,
            sampling_rate=sampling_rate,
            transforms=transforms
        )

    @staticmethod
    def from_dict(data: dict) -> 'Recording':
        raw_sources = data.pop('sources')
        return Recording(sources=[AudioSource.from_dict(s) for s in raw_sources], **data)


class RecordingSet(Serializable, Sequence[Recording]):
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
            >>> recs_24k = recs.resample(24000)
    """

    def __init__(self, recordings: Mapping[str, Recording] = None) -> None:
        self.recordings = ifnone(recordings, {})

    def __eq__(self, other: 'RecordingSet') -> bool:
        return self.recordings == other.recordings

    @property
    def is_lazy(self) -> bool:
        """
        Indicates whether this manifest was opened in lazy (read-on-the-fly) mode or not.
        """
        from lhotse.serialization import LazyJsonlIterator
        return isinstance(self.recordings, LazyJsonlIterator)

    @property
    def ids(self) -> Iterable[str]:
        return self.recordings.keys()

    @staticmethod
    def from_recordings(recordings: Iterable[Recording]) -> 'RecordingSet':
        return RecordingSet(recordings=index_by_id_and_check(recordings))

    @staticmethod
    def from_dir(
            path: Pathlike,
            pattern: str,
            num_jobs: int = 1,
            force_opus_sampling_rate: Optional[int] = None,
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
        :return: a new ``Recording`` instance pointing to the audio file.
        """
        msg = f'Scanning audio files ({pattern})'
        fn = Recording.from_file
        if force_opus_sampling_rate is not None:
            fn = partial(Recording.from_file, force_opus_sampling_rate=force_opus_sampling_rate)
        if num_jobs == 1:
            # Avoid spawning process for one job.
            return RecordingSet.from_recordings(
                tqdm(
                    map(fn, Path(path).rglob(pattern)),
                    desc=msg
                )
            )
        with ProcessPoolExecutor(num_jobs) as ex:
            return RecordingSet.from_recordings(
                tqdm(
                    ex.map(fn, Path(path).rglob(pattern)),
                    desc=msg
                )
            )

    @staticmethod
    def from_dicts(data: Iterable[dict]) -> 'RecordingSet':
        return RecordingSet.from_recordings(Recording.from_dict(raw_rec) for raw_rec in data)

    def to_dicts(self) -> Iterable[dict]:
        return (r.to_dict() for r in self)

    def filter(self, predicate: Callable[[Recording], bool]) -> 'RecordingSet':
        """
        Return a new RecordingSet with the Recordings that satisfy the `predicate`.

        :param predicate: a function that takes a recording as an argument and returns bool.
        :return: a filtered RecordingSet.
        """
        return RecordingSet.from_recordings(rec for rec in self if predicate(rec))

    def shuffle(self, rng: Optional[random.Random] = None) -> 'RecordingSet':
        """
        Shuffle the recording IDs in the current :class:`.RecordingSet` and return a shuffled copy of self.

        :param rng: an optional instance of ``random.Random`` for precise control of randomness.
        :return: a shuffled copy of self.
        """
        if rng is None:
            rng = random
        ids = list(self.ids)
        rng.shuffle(ids)
        return RecordingSet(recordings={rid: self[rid] for rid in ids})

    def split(self, num_splits: int, shuffle: bool = False, drop_last: bool = False) -> List['RecordingSet']:
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
            RecordingSet.from_recordings(subset) for subset in
            split_sequence(self, num_splits=num_splits, shuffle=shuffle, drop_last=drop_last)
        ]

    def subset(self, first: Optional[int] = None, last: Optional[int] = None) -> 'RecordingSet':
        """
        Return a new ``RecordingSet`` according to the selected subset criterion.
        Only a single argument to ``subset`` is supported at this time.

        :param first: int, the number of first recordings to keep.
        :param last: int, the number of last recordings to keep.
        :return: a new ``RecordingSet`` with the subset results.
        """
        assert exactly_one_not_null(first, last), "subset() can handle only one non-None arg."

        if first is not None:
            assert first > 0
            if first > len(self):
                logging.warning(f'RecordingSet has only {len(self)} items but first {first} required; '
                                f'not doing anything.')
                return self
            return RecordingSet.from_recordings(islice(self, first))

        if last is not None:
            assert last > 0
            if last > len(self):
                logging.warning(f'RecordingSet has only {len(self)} items but last {last} required; '
                                f'not doing anything.')
                return self
            return RecordingSet.from_recordings(islice(self, len(self) - last, len(self)))

    def load_audio(
            self,
            recording_id: str,
            channels: Optional[Channels] = None,
            offset_seconds: float = 0.0,
            duration_seconds: Optional[float] = None,
    ) -> np.ndarray:
        return self.recordings[recording_id].load_audio(
            channels=channels,
            offset=offset_seconds,
            duration=duration_seconds
        )

    def with_path_prefix(self, path: Pathlike) -> 'RecordingSet':
        return RecordingSet.from_recordings(r.with_path_prefix(path) for r in self)

    def num_channels(self, recording_id: str) -> int:
        return self.recordings[recording_id].num_channels

    def sampling_rate(self, recording_id: str) -> int:
        return self.recordings[recording_id].sampling_rate

    def num_samples(self, recording_id: str) -> int:
        return self.recordings[recording_id].num_samples

    def duration(self, recording_id: str) -> Seconds:
        return self.recordings[recording_id].duration

    def perturb_speed(self, factor: float, affix_id: bool = True) -> 'RecordingSet':
        """
        Return a new ``RecordingSet`` that will lazily perturb the speed while loading audio.
        The ``num_samples`` and ``duration`` fields are updated to reflect the
        shrinking/extending effect of speed.

        :param factor: The speed will be adjusted this many times (e.g. factor=1.1 means 1.1x faster).
        :param affix_id: When true, we will modify the ``Recording.id`` field
            by affixing it with "_sp{factor}".
        :return: a ``RecordingSet`` containing the perturbed ``Recording`` objects.
        """
        return RecordingSet.from_recordings(r.perturb_speed(factor=factor, affix_id=affix_id) for r in self)

    def perturb_tempo(self, factor: float, affix_id: bool = True) -> 'RecordingSet':
        """
        Return a new ``RecordingSet`` that will lazily perturb the tempo while loading audio.
        The ``num_samples`` and ``duration`` fields are updated to reflect the
        shrinking/extending effect of tempo.

        :param factor: The speed will be adjusted this many times (e.g. factor=1.1 means 1.1x faster).
        :param affix_id: When true, we will modify the ``Recording.id`` field
            by affixing it with "_sp{factor}".
        :return: a ``RecordingSet`` containing the perturbed ``Recording`` objects.
        """
        return RecordingSet.from_recordings(r.perturb_tempo(factor=factor, affix_id=affix_id) for r in self)

    def perturb_volume(self, factor: float, affix_id: bool = True) -> 'RecordingSet':
        """
        Return a new ``RecordingSet`` that will lazily perturb the volume while loading audio.

        :param factor: The volume scale to be applied (e.g. factor=1.1 means 1.1x louder).
        :param affix_id: When true, we will modify the ``Recording.id`` field
            by affixing it with "_sp{factor}".
        :return: a ``RecordingSet`` containing the perturbed ``Recording`` objects.
        """
        return RecordingSet.from_recordings(r.perturb_volume(factor=factor, affix_id=affix_id) for r in self)

    def resample(self, sampling_rate: int) -> 'RecordingSet':
        """
        Apply resampling to all recordings in the ``RecordingSet`` and return a new ``RecordingSet``.
        :param sampling_rate: The new sampling rate.
        :return: a new ``RecordingSet`` with lazily resampled ``Recording`` objects.
        """
        return RecordingSet.from_recordings(r.resample(sampling_rate) for r in self)

    def __repr__(self) -> str:
        return f'RecordingSet(len={len(self)})'

    def __contains__(self, item: Union[str, Recording]) -> bool:
        if isinstance(item, str):
            return item in self.recordings
        else:
            return item.id in self.recordings

    def __getitem__(self, recording_id_or_index: Union[int, str]) -> Recording:
        if isinstance(recording_id_or_index, str):
            return self.recordings[recording_id_or_index]
        # ~100x faster than list(dict.values())[index] for 100k elements
        return next(val for idx, val in enumerate(self.recordings.values()) if idx == recording_id_or_index)

    def __iter__(self) -> Iterable[Recording]:
        return iter(self.recordings.values())

    def __len__(self) -> int:
        return len(self.recordings)

    def __add__(self, other: 'RecordingSet') -> 'RecordingSet':
        return RecordingSet(recordings={**self.recordings, **other.recordings})


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
    """

    def __init__(self, base_audio: np.ndarray, sampling_rate: int):
        """
        :param base_audio: A numpy array with the audio samples for the base signal
            (all the other signals will be mixed to it).
        :param sampling_rate: Sampling rate of the audio.
        """
        self.tracks = [base_audio]
        self.sampling_rate = sampling_rate
        self.reference_energy = audio_energy(base_audio)
        self.dtype = self.tracks[0].dtype

    @property
    def unmixed_audio(self) -> np.ndarray:
        """
        Return a numpy ndarray with the shape (num_tracks, num_samples), where each track is
        zero padded and scaled adequately to the offsets and SNR used in ``add_to_mix`` call.
        """
        return np.vstack(self.tracks)

    @property
    def mixed_audio(self) -> np.ndarray:
        """
        Return a numpy ndarray with the shape (1, num_samples) - a mono mix of the tracks
        supplied with ``add_to_mix`` calls.
        """
        return np.sum(self.unmixed_audio, axis=0, keepdims=True)

    def add_to_mix(
            self,
            audio: np.ndarray,
            snr: Optional[Decibels] = None,
            offset: Seconds = 0.0,
    ):
        """
        Add audio (only support mono-channel) of a new track into the mix.
        :param audio: An array of audio samples to be mixed in.
        :param snr: Signal-to-noise ratio, assuming `audio` represents noise (positive SNR - lower `audio` energy,
        negative SNR - higher `audio` energy)
        :param offset: How many seconds to shift `audio` in time. For mixing, the signal will be padded before
        the start with low energy values.
        :return:
        """
        assert audio.shape[0] == 1  # TODO: support multi-channels
        assert offset >= 0.0, "Negative offset in mixing is not supported."

        reference_audio = self.tracks[0]
        num_samples_offset = round(offset * self.sampling_rate)
        current_num_samples = reference_audio.shape[1]

        audio_to_add = audio

        # When there is an offset, we need to pad before the start of the audio we're adding.
        if offset > 0:
            audio_to_add = np.hstack([
                np.zeros((1, num_samples_offset), self.dtype),
                audio_to_add
            ])

        incoming_num_samples = audio_to_add.shape[1]
        mix_num_samples = max(current_num_samples, incoming_num_samples)

        # When the existing samples are less than what we anticipate after the mix,
        # we need to pad after the end of the existing audio mixed so far.
        # Since we're keeping every track as a separate entry in the ``self.tracks`` list,
        # we need to pad each of them so that their shape matches when performing the final mix.
        if current_num_samples < mix_num_samples:
            for idx in range(len(self.tracks)):
                padded_audio = np.hstack([
                    self.tracks[idx],
                    np.zeros((1, mix_num_samples - current_num_samples), self.dtype)
                ])
                self.tracks[idx] = padded_audio

        # When the audio we're mixing in are shorter that the anticipated mix length,
        # we need to pad after their end.
        # Note: we're doing that non-efficiently, as it we potentially re-allocate numpy arrays twice,
        # during this padding and the  offset padding before. If that's a bottleneck, we'll optimize.
        if incoming_num_samples < mix_num_samples:
            audio_to_add = np.hstack([
                audio_to_add,
                np.zeros((1, mix_num_samples - incoming_num_samples), self.dtype)
            ])

        # When SNR is requested, find what gain is needed to satisfy the SNR
        gain = 1.0
        if snr is not None:
            added_audio_energy = audio_energy(audio)
            if added_audio_energy <= 0.0:
                raise NonPositiveEnergyError(
                    f"To perform mix, energy must be non-zero and non-negative (got {added_audio_energy}). "
                )
            target_energy = self.reference_energy * (10.0 ** (-snr / 10))
            # When mixing time-domain signals, we are working with root-power (field) quantities,
            # whereas the energy ratio applies to power quantities. To compute the gain correctly,
            # we need to take a square root of the energy ratio.
            gain = sqrt(target_energy / added_audio_energy)

        # self.mixed_audio = reference_audio + gain * audio_to_add
        self.tracks.append(gain * audio_to_add)


def audio_energy(audio: np.ndarray) -> float:
    return float(np.average(audio ** 2))


FileObject = Any  # Alias for file-like objects


def read_audio(
        path_or_fd: Union[Pathlike, FileObject],
        offset: Seconds = 0.0,
        duration: Optional[Seconds] = None,
        force_opus_sampling_rate: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    if isinstance(path_or_fd, (str, Path)) and str(path_or_fd).lower().endswith('.opus'):
        return read_opus(
            path_or_fd,
            offset=offset,
            duration=duration,
            force_opus_sampling_rate=force_opus_sampling_rate,
        )
    elif isinstance(path_or_fd, (str, Path)) and str(path_or_fd).lower().endswith('.sph'):
        return read_sph(
            path_or_fd,
            offset=offset,
            duration=duration
        )
    try:
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
            return sf_desc.read(frames=frame_duration, dtype=np.float32, always_2d=False).T, sampling_rate
    except:
        return _audioread_load(path_or_fd, offset=offset, duration=duration)


class LibsndfileCompatibleAudioInfo(NamedTuple):
    channels: int
    frames: int
    samplerate: int
    duration: float


def audioread_info(path: Pathlike) -> LibsndfileCompatibleAudioInfo:
    """
    Return an audio info data structure that's a compatible subset of ``pysoundfile.info()``
    that we need to create a ``Recording`` manifest.
    """
    import audioread

    # We just read the file and compute the number of samples
    # -- no other method seems fully reliable...
    with audioread.audio_open(path, backends=_available_audioread_backends()) as input_file:
        shape = _audioread_load(input_file)[0].shape
        if len(shape) == 1:
            num_samples = shape[0]
        else:
            num_samples = shape[1]
        return LibsndfileCompatibleAudioInfo(
            channels=input_file.channels,
            frames=num_samples,
            samplerate=input_file.samplerate,
            duration=num_samples / input_file.samplerate
        )


@lru_cache(maxsize=1)
def _available_audioread_backends():
    """
    Reduces the overhead of ``audioread.audio_open()`` when called repeatedly
    by caching the results of scanning for FFMPEG etc.
    """
    import audioread
    backends = audioread.available_backends()
    logging.info(f'Using audioread. Available backends: {backends}')
    return backends


def _audioread_load(
        path_or_file: Union[Pathlike, FileObject],
        offset: Seconds = 0.0,
        duration: Seconds = None,
        dtype=np.float32
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
            yield audioread.audio_open(path_or_file, backends=_available_audioread_backends())
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
                frame = frame[(s_start - n_prev):]

            # tack on the current frame
            y.append(frame)

    if y:
        y = np.concatenate(y)
        if n_channels > 1:
            y = y.reshape((-1, n_channels)).T
    else:
        y = np.empty(0, dtype=dtype)

    return y, sr_native


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


# This constant defines by how much our estimation can be mismatched with
# the actual number of samples after applying audio augmentation.
# Chains of augmentation effects (such as resampling, speed perturb) can cause
# difficult to predict roundings and return a few samples more/less than we estimate.
# The default tolerance is a quarter of a millisecond
# (the actual number of samples is computed based on the sampling rate).
AUGMENTATION_DURATION_TOLERANCE: Seconds = 0.00025


def assert_and_maybe_fix_num_samples(
        audio: np.ndarray,
        offset: Seconds,
        duration: Optional[Seconds],
        recording: Recording
) -> np.ndarray:
    # When resampling in high sampling rates (48k -> 44.1k)
    # it is difficult to estimate how sox will perform rounding;
    # we will just add/remove one sample to be consistent with
    # what we have estimated.
    # This effect is exacerbated by chaining multiple augmentations together.
    expected_num_samples = compute_num_samples(
        duration=duration if duration is not None else recording.duration - offset,
        sampling_rate=recording.sampling_rate
    )
    diff = expected_num_samples - audio.shape[1]
    if diff == 0:
        return audio  # this is normal condition
    allowed_diff = int(ceil(AUGMENTATION_DURATION_TOLERANCE * recording.sampling_rate))
    if 0 < diff <= allowed_diff:
        # note the extra colon in -1:, which preserves the shape
        audio = np.append(audio, audio[:, -diff:], axis=1)
        return audio
    elif -allowed_diff <= diff < 0:
        audio = audio[:, :diff]
        return audio
    else:
        raise ValueError("The number of declared samples in the recording diverged from the one obtained "
                         f"when loading audio (offset={offset}, duration={duration}). "
                         f"This could be internal Lhotse's error or a faulty transform implementation. "
                         "Please report this issue in Lhotse and show the "
                         f"following: diff={diff}, audio.shape={audio.shape}, recording={recording}")


def opus_info(
        path: Pathlike,
        force_opus_sampling_rate: Optional[int] = None
) -> LibsndfileCompatibleAudioInfo:
    samples, sampling_rate = read_opus(path, force_opus_sampling_rate=force_opus_sampling_rate)
    return LibsndfileCompatibleAudioInfo(
        channels=samples.shape[0],
        frames=samples.shape[1],
        samplerate=sampling_rate,
        duration=samples.shape[1] / sampling_rate
    )


def read_opus(
        path: Pathlike,
        offset: Seconds = 0.0,
        duration: Optional[Seconds] = None,
        force_opus_sampling_rate: Optional[int] = None
) -> Tuple[np.ndarray, int]:
    """
    Reads OPUS files using ffmpeg in a shell subprocess.
    Unlike audioread, correctly supports offsets and durations for reading short chunks.
    Optionally, we can force ffmpeg to resample to the true sampling rate (if we know it up-front).

    :return: a tuple of audio samples and the sampling rate.
    """
    # Construct the ffmpeg command depending on the arguments passed.
    cmd = f'ffmpeg'
    sampling_rate = 48000
    # Note: we have to add offset and duration options (-ss and -t) BEFORE specifying the input
    #       (-i), otherwise ffmpeg will decode everything and trim afterwards...
    if offset > 0:
        cmd += f' -ss {offset}'
    if duration is not None:
        cmd += f' -t {duration}'
    # Add the input specifier after offset and duration.
    cmd += f' -i {path}'
    # Optionally resample the output.
    if force_opus_sampling_rate is not None:
        cmd += f' -ar {force_opus_sampling_rate}'
        sampling_rate = force_opus_sampling_rate
    # Read audio samples directly as float32.
    cmd += ' -f f32le pipe:1'
    # Actual audio reading.
    proc = run(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    raw_audio = proc.stdout
    audio = np.frombuffer(raw_audio, dtype=np.float32)
    # Determine if the recording is mono or stereo and decode accordingly.
    channel_string = parse_channel_from_ffmpeg_output(proc.stderr)
    if channel_string == 'stereo':
        new_audio = np.empty((2, audio.shape[0] // 2), dtype=np.float32)
        new_audio[0, :] = audio[::2]
        new_audio[1, :] = audio[1::2]
        audio = new_audio
    elif channel_string == 'mono':
        audio = audio.reshape(1, -1)
    else:
        raise NotImplementedError(f'Unknown channel description from ffmpeg: {channel_string}')
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
        samplerate=sampling_rate,
        duration=samples.shape[1] / sampling_rate
    )


def read_sph(
        sph_path: Pathlike,
        offset: Seconds = 0.0,
        duration: Optional[Seconds] = None
) -> Tuple[np.ndarray, int]:
    """
    Reads SPH files using sph2pipe in a shell subprocess.
    Unlike audioread, correctly supports offsets and durations for reading short chunks.

    :return: a tuple of audio samples and the sampling rate.
    """

    sph_path = Path(sph_path)

    # Construct the sph2pipe command depending on the arguments passed.
    cmd = f'sph2pipe -f wav -p -t {offset}:'

    if duration is not None:
        cmd += f'{round(offset + duration, 5)}'
    # Add the input specifier after offset and duration.
    cmd += f' {sph_path}'

    # Actual audio reading.
    proc = BytesIO(run(cmd, shell=True, check=True, stdout=PIPE, stderr=PIPE).stdout)

    import soundfile as sf
    with sf.SoundFile(proc) as sf_desc:
        audio, sampling_rate = sf_desc.read(dtype=np.float32), sf_desc.samplerate
        audio = audio.reshape(1, -1) if sf_desc.channels == 1 else audio.T

    return audio, sampling_rate
