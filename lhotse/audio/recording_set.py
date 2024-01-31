import logging
import random
import re
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import islice
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Union

import numpy as np
from tqdm.asyncio import tqdm

from lhotse.audio.recording import Channels, Recording
from lhotse.lazy import AlgorithmMixin
from lhotse.serialization import Serializable
from lhotse.utils import (
    Pathlike,
    Seconds,
    exactly_one_not_null,
    ifnone,
    split_manifest_lazy,
    split_sequence,
)


class RecordingSet(Serializable, AlgorithmMixin):
    """
    :class:`~lhotse.audio.RecordingSet` represents a collection of recordings.
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

    def __init__(self, recordings: Optional[Iterable[Recording]] = None) -> None:
        self.recordings = ifnone(recordings, {})

    def __eq__(self, other: "RecordingSet") -> bool:
        return self.recordings == other.recordings

    @property
    def data(self) -> Union[Dict[str, Recording], Iterable[Recording]]:
        """Alias property for ``self.recordings``"""
        return self.recordings

    @property
    def ids(self) -> Iterable[str]:
        return (r.id for r in self)

    @staticmethod
    def from_recordings(recordings: Iterable[Recording]) -> "RecordingSet":
        return RecordingSet(list(recordings))

    from_items = from_recordings

    @staticmethod
    def from_dir(
        path: Pathlike,
        pattern: str,
        num_jobs: int = 1,
        force_opus_sampling_rate: Optional[int] = None,
        recording_id: Optional[Callable[[Path], str]] = None,
        exclude_pattern: Optional[str] = None,
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
        :param exclude_pattern: optional regex string for identifying file name patterns to exclude.
            There has to be a full regex match to trigger exclusion.
        :return: a new ``Recording`` instance pointing to the audio file.
        """
        msg = f"Scanning audio files ({pattern})"

        file_read_worker = partial(
            Recording.from_file,
            force_opus_sampling_rate=force_opus_sampling_rate,
            recording_id=recording_id,
        )

        it = Path(path).rglob(pattern)
        if exclude_pattern is not None:
            exclude_pattern = re.compile(exclude_pattern)
            it = filter(lambda p: exclude_pattern.match(p.name) is None, it)

        if num_jobs == 1:
            # Avoid spawning process for one job.
            return RecordingSet.from_recordings(
                tqdm(map(file_read_worker, it), desc=msg)
            )
        with ProcessPoolExecutor(num_jobs) as ex:
            return RecordingSet.from_recordings(
                tqdm(
                    ex.map(file_read_worker, it),
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
            return out

        if last is not None:
            assert last > 0
            if last > len(self):
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
        return self[recording_id].load_audio(
            channels=channels, offset=offset_seconds, duration=duration_seconds
        )

    def with_path_prefix(self, path: Pathlike) -> "RecordingSet":
        return RecordingSet.from_recordings(r.with_path_prefix(path) for r in self)

    def num_channels(self, recording_id: str) -> int:
        return self[recording_id].num_channels

    def sampling_rate(self, recording_id: str) -> int:
        return self[recording_id].sampling_rate

    def num_samples(self, recording_id: str) -> int:
        return self[recording_id].num_samples

    def duration(self, recording_id: str) -> Seconds:
        return self[recording_id].duration

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

    def __getitem__(self, index_or_id: Union[int, str]) -> Recording:
        try:
            return self.recordings[index_or_id]  # int passed, eager manifest, fast
        except TypeError:
            # either lazy manifest or str passed, both are slow
            if self.is_lazy:
                return next(item for idx, item in enumerate(self) if idx == index_or_id)
            else:
                # string id passed, support just for backward compatibility, not recommended
                return next(item for item in self if item.id == index_or_id)

    def __contains__(self, other: Union[str, Recording]) -> bool:
        if isinstance(other, str):
            return any(other == item.id for item in self)
        else:
            return any(other.id == item.id for item in self)

    def __iter__(self) -> Iterable[Recording]:
        yield from self.recordings

    def __len__(self) -> int:
        return len(self.recordings)
