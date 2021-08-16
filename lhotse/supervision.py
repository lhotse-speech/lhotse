import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from itertools import groupby, islice
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Union

from lhotse.serialization import Serializable
from lhotse.utils import Pathlike, Seconds, TimeSpan, asdict_nonull, compute_num_samples, exactly_one_not_null, \
    fastcopy, ifnone, index_by_id_and_check, overspans, perturb_num_samples, split_sequence


@dataclass
class AlignmentItem:
    """
    This class contains an alignment item, for example a word, along with its
    start time (w.r.t. the start of recording) and duration. It can potentially
    be used to store other kinds of alignment items, such as subwords, pdfid's etc.
    
    We use dataclasses instead of namedtuples (even though they are potentially slower)
    because of a serialization bug in nested namedtuples and dataclasses in Python 3.7
    (see this: https://alexdelorenzo.dev/programming/2018/08/09/bug-in-dataclass.html).
    We can revert to namedtuples if we bump up the Python requirement to 3.8+.
    """
    symbol: str
    start: Seconds
    duration: Seconds

    @property
    def end(self) -> Seconds:
        return round(self.start + self.duration, ndigits=8)

    def with_offset(self, offset: Seconds) -> 'AlignmentItem':
        """Return an identical ``AlignmentItem``, but with the ``offset`` added to the ``start`` field."""
        return AlignmentItem(self.symbol, round(self.start + offset, ndigits=8), self.duration)

    def perturb_speed(self, factor: float, sampling_rate: int) -> 'AlignmentItem':
        """
        Return an ``AlignmentItem`` that has time boundaries matching the
        recording/cut perturbed with the same factor. See :meth:`SupervisionSegment.perturb_speed` 
        for details.
        """
        start_sample = compute_num_samples(self.start, sampling_rate)
        num_samples = compute_num_samples(self.duration, sampling_rate)
        new_start = perturb_num_samples(start_sample, factor) / sampling_rate
        new_duration = perturb_num_samples(num_samples, factor) / sampling_rate
        return AlignmentItem(self.symbol, new_start, new_duration)

    def trim(self, end: Seconds, start: Seconds = 0) -> 'AlignmentItem':
        """
        See :met:`SupervisionSegment.trim`.
        """
        assert start >= 0
        start_exceeds_by = abs(min(0, self.start - start))
        end_exceeds_by = max(0, self.end - end)
        return AlignmentItem(self.symbol, max(start, self.start), self.duration - end_exceeds_by - start_exceeds_by)

    def transform(self, transform_fn: Callable[[str], str]) -> 'AlignmentItem':
        """
        Perform specified transformation on the alignment content.
        """
        return AlignmentItem(transform_fn(self.symbol), self.start, self.duration)


@dataclass
class SupervisionSegment:
    """
    :class:`~lhotse.supervsion.SupervisionSegment` represents a time interval (segment) annotated with some
    supervision labels and/or metadata, such as the transcription, the speaker identity, the language, etc.

    Each supervision has unique ``id`` and always refers to a specific recording (via ``recording_id``)
    and a specific ``channel`` (by default, 0).
    It's also characterized by the start time (relative to the beginning of a :class:`~lhotse.audio.Recording`
    or a :class:`~lhotse.cut.Cut`) and a duration, both expressed in seconds.

    The remaining fields are all optional, and their availability depends on specific corpora.
    Since it is difficult to predict all possible types of metadata, the ``custom`` field (a dict) can be used to
    insert types of supervisions that are not supported out of the box.

    :class:`~lhotse.supervsion.SupervisionSegment` may contain multiple types of alignments.
    The ``alignment`` field is a dict, indexed by alignment's type (e.g., ``word`` or ``phone``),
    and contains a list of :class:`~lhotse.supervision.AlignmentItem` objects -- simple structures
    that contain a given symbol and its time interval.
    Alignments can be read from CTM files or created programatically.

    Examples

        A simple segment with no supervision information::

            >>> from lhotse import SupervisionSegment
            >>> sup0 = SupervisionSegment(
            ...     id='rec00001-sup00000', recording_id='rec00001',
            ...     start=0.5, duration=5.0, channel=0
            ... )

        Typical supervision containing transcript, speaker ID, gender, and language::

            >>> sup1 = SupervisionSegment(
            ...     id='rec00001-sup00001', recording_id='rec00001',
            ...     start=5.5, duration=3.0, channel=0,
            ...     text='transcript of the second segment',
            ...     speaker='Norman Dyhrentfurth', language='English', gender='M'
            ... )

        Two supervisions denoting overlapping speech on two separate channels in a microphone array/multiple headsets
        (pay attention to ``start``, ``duration``, and ``channel``)::

            >>> sup2 = SupervisionSegment(
            ...     id='rec00001-sup00002', recording_id='rec00001',
            ...     start=15.0, duration=5.0, channel=0,
            ...     text="i have incredibly good news for you",
            ...     speaker='Norman Dyhrentfurth', language='English', gender='M'
            ... )
            >>> sup3 = SupervisionSegment(
            ...     id='rec00001-sup00003', recording_id='rec00001',
            ...     start=18.0, duration=3.0, channel=1,
            ...     text="say what",
            ...     speaker='Hervey Arman', language='English', gender='M'
            ... )

        A supervision with a phone alignment::

            >>> from lhotse.supervision import AlignmentItem
            >>> sup4 = SupervisionSegment(
            ...     id='rec00001-sup00004', recording_id='rec00001',
            ...     start=33.0, duration=1.0, channel=0,
            ...     text="ice",
            ...     speaker='Maryla Zechariah', language='English', gender='F'
            ...     alignment={
            ...         'phone': [
            ...             AlignmentItem(symbol='AY0', start=33.0, duration=0.6),
            ...             AlignmentItem(symbol='S', start=33.6, duration=0.4)
            ...         ]
            ...     }
            ... )

        Converting :class:`~lhotse.supervsion.SupervisionSegment` to a ``dict``::

            >>> sup0.to_dict()
            {'id': 'rec00001-sup00000', 'recording_id': 'rec00001', 'start': 0.5, 'duration': 5.0, 'channel': 0}


    """
    id: str
    recording_id: str
    start: Seconds
    duration: Seconds
    channel: int = 0
    text: Optional[str] = None
    language: Optional[str] = None
    speaker: Optional[str] = None
    gender: Optional[str] = None
    custom: Optional[Dict[str, Any]] = None
    alignment: Optional[Dict[str, List[AlignmentItem]]] = None

    @property
    def end(self) -> Seconds:
        return round(self.start + self.duration, ndigits=8)

    def with_offset(self, offset: Seconds) -> 'SupervisionSegment':
        """Return an identical ``SupervisionSegment``, but with the ``offset`` added to the ``start`` field."""
        return SupervisionSegment(
            id=self.id,
            recording_id=self.recording_id,
            start=round(self.start + offset, ndigits=8),
            duration=self.duration,
            channel=self.channel,
            text=self.text,
            language=self.language,
            speaker=self.speaker,
            gender=self.gender,
            custom=self.custom,
            alignment={
                type: [
                    item.with_offset(offset=offset)
                    for item in ali
                ]
                for type, ali in self.alignment.items()
            } if self.alignment else None
        )

    def perturb_speed(
            self,
            factor: float,
            sampling_rate: int,
            affix_id: bool = True
    ) -> 'SupervisionSegment':
        """
        Return a ``SupervisionSegment`` that has time boundaries matching the
        recording/cut perturbed with the same factor.

        :param factor: The speed will be adjusted this many times (e.g. factor=1.1 means 1.1x faster).
        :param sampling_rate: The sampling rate is necessary to accurately perturb the start
            and duration (going through the sample counts).
        :param affix_id: When true, we will modify the ``id`` and ``recording_id`` fields
            by affixing it with "_sp{factor}".
        :return: a modified copy of the current ``Recording``.
        """
        start_sample = compute_num_samples(self.start, sampling_rate)
        num_samples = compute_num_samples(self.duration, sampling_rate)
        new_start = perturb_num_samples(start_sample, factor) / sampling_rate
        new_duration = perturb_num_samples(num_samples, factor) / sampling_rate
        return fastcopy(
            self,
            id=f'{self.id}_sp{factor}' if affix_id else self.id,
            recording_id=f'{self.recording_id}_sp{factor}' if affix_id else self.id,
            start=new_start,
            duration=new_duration,
            alignment={
                type:[
                    item.perturb_speed(factor=factor, sampling_rate=sampling_rate)
                    for item in ali
                ]
                for type, ali in self.alignment.items()
            } if self.alignment else None
        )

    def perturb_tempo(
            self,
            factor: float,
            sampling_rate: int,
            affix_id: bool = True
    ) -> 'SupervisionSegment':
        """
        Return a ``SupervisionSegment`` that has time boundaries matching the
        recording/cut perturbed with the same factor.

        :param factor: The tempo will be adjusted this many times (e.g. factor=1.1 means 1.1x faster).
        :param sampling_rate: The sampling rate is necessary to accurately perturb the start
            and duration (going through the sample counts).
        :param affix_id: When true, we will modify the ``id`` and ``recording_id`` fields
            by affixing it with "_tp{factor}".
        :return: a modified copy of the current ``Recording``.
        """

        # speed and tempo perturbation have the same effect on supervisions
        perturbed = self.perturb_speed(factor, sampling_rate, affix_id)
        return fastcopy(
            perturbed,
            id=f'{self.id}_tp{factor}' if affix_id else self.id,
            recording_id=f'{self.recording_id}_tp{factor}' if affix_id else self.id,
        )

    def trim(self, end: Seconds, start: Seconds = 0) -> 'SupervisionSegment':
        """
        Return an identical ``SupervisionSegment``, but ensure that ``self.start`` is not negative (in which case
        it's set to 0) and ``self.end`` does not exceed the ``end`` parameter. If a `start` is optionally
        provided, the supervision is trimmed from the left (note that start should be relative to the cut times).

        This method is useful for ensuring that the supervision does not exceed a cut's bounds,
        in which case pass ``cut.duration`` as the ``end`` argument, since supervision times are relative to the cut.
        """
        assert start >= 0
        start_exceeds_by = abs(min(0, self.start - start))
        end_exceeds_by = max(0, self.end - end)
        return fastcopy(
            self,
            start=max(start, self.start),
            duration=self.duration - end_exceeds_by - start_exceeds_by,
            alignment={
                type:[
                    item.trim(end=end, start=start)
                    for item in ali
                ]
                for type, ali in self.alignment.items()
            } if self.alignment else None
        )

    def map(self, transform_fn: Callable[['SupervisionSegment'], 'SupervisionSegment']) -> 'SupervisionSegment':
        """
        Return a copy of the current segment, transformed with ``transform_fn``.

        :param transform_fn: a function that takes a segment as input, transforms it and returns a new segment.
        :return: a modified ``SupervisionSegment``.
        """
        return transform_fn(self)

    def transform_text(self, transform_fn: Callable[[str], str]) -> 'SupervisionSegment':
        """
        Return a copy of the current segment with transformed ``text`` field.
        Useful for text normalization, phonetic transcription, etc.

        :param transform_fn: a function that accepts a string and returns a string.
        :return: a ``SupervisionSegment`` with adjusted text.
        """
        if self.text is None:
            return self
        return fastcopy(self, text=transform_fn(self.text))
    
    def transform_alignment(
        self,
        transform_fn: Callable[[str], str],
        type: Optional[str] = 'word'
    ) -> 'SupervisionSegment':
        """
        Return a copy of the current segment with transformed ``alignment`` field.
        Useful for text normalization, phonetic transcription, etc.

        :param type:  alignment type to transform (key for alignment dict).
        :param transform_fn: a function that accepts a string and returns a string.
        :return: a ``SupervisionSegment`` with adjusted alignments.
        """
        if self.alignment is None:
            return self
        return fastcopy(
            self,
            alignment={
                ali_type:[
                    item.transform(transform_fn=transform_fn)
                    if ali_type == type else item
                    for item in ali
                ]
                for ali_type, ali in self.alignment.items()
            }
        )

    def to_dict(self) -> dict:
        return asdict_nonull(self)

    @staticmethod
    def from_dict(data: dict) -> 'SupervisionSegment':
        return SupervisionSegment(
            **{
                key:(
                    {k:[AlignmentItem(**x) for x in v] for k,v in value.items()}
                    if key == 'alignment' else value)
                for key, value in data.items()
            }
        )


class SupervisionSet(Serializable, Sequence[SupervisionSegment]):
    """
    :class:`~lhotse.supervision.SupervisionSet` represents a collection of segments containing some
    supervision information (see :class:`~lhotse.supervision.SupervisionSegment`),
    that are indexed by segment IDs.

    It acts as a Python ``dict``, extended with an efficient ``find`` operation that indexes and caches
    the supervision segments in an interval tree.
    It allows to quickly find supervision segments that correspond to a specific time interval.

    When coming from Kaldi, think of :class:`~lhotse.supervision.SupervisionSet` as a ``segments`` file on steroids,
    that may also contain *text*, *utt2spk*, *utt2gender*, *utt2dur*, etc.

    Examples

        Building a :class:`~lhotse.supervision.SupervisionSet`::

            >>> from lhotse import SupervisionSet, SupervisionSegment
            >>> sups = SupervisionSet.from_segments([SupervisionSegment(...), ...])

        Writing/reading a :class:`~lhotse.supervision.SupervisionSet`::

            >>> sups.to_file('supervisions.jsonl.gz')
            >>> sups2 = SupervisionSet.from_file('supervisions.jsonl.gz')

        Using :class:`~lhotse.supervision.SupervisionSet` like a dict::

            >>> 'rec00001-sup00000' in sups
            True
            >>> sups['rec00001-sup00000']
            SupervisionSegment(id='rec00001-sup00000', recording_id='rec00001', start=0.5, ...)
            >>> for segment in sups:
            ...     pass

        Searching by ``recording_id`` and time interval::

            >>> matched_segments = sups.find(recording_id='rec00001', start_after=17.0, end_before=25.0)

        Manipulation::

            >>> longer_than_5s = sups.filter(lambda s: s.duration > 5)
            >>> first_100 = sups.subset(first=100)
            >>> split_into_4 = sups.split(num_splits=4)
            >>> shuffled = sups.shuffle()
    """

    def __init__(self, segments: Mapping[str, SupervisionSegment]) -> None:
        self.segments = ifnone(segments, {})

    def __eq__(self, other: 'SupervisionSet') -> bool:
        return self.segments == other.segments

    @property
    def is_lazy(self) -> bool:
        """
        Indicates whether this manifest was opened in lazy (read-on-the-fly) mode or not.
        """
        from lhotse.serialization import LazyJsonlIterator
        return isinstance(self.segments, LazyJsonlIterator)

    @property
    def ids(self) -> Iterable[str]:
        return self.segments.keys()

    @staticmethod
    def from_segments(segments: Iterable[SupervisionSegment]) -> 'SupervisionSet':
        return SupervisionSet(segments=index_by_id_and_check(segments))

    @staticmethod
    def from_dicts(data: Iterable[Dict]) -> 'SupervisionSet':
        return SupervisionSet.from_segments(SupervisionSegment.from_dict(s) for s in data)

    def with_alignment_from_ctm(
            self,
            ctm_file: Pathlike,
            type: str = 'word',
            match_channel: bool = False
    ) -> 'SupervisionSet':
        """
        Add alignments from CTM file to the supervision set.
        
        :param ctm: Path to CTM file.
        :param type: Alignment type (optional, default = `word`).
        :param match_channel: if True, also match channel between CTM and SupervisionSegment
        :return: A new SupervisionSet with AlignmentItem objects added to the segments.
        """
        ctm_words = []
        with open(ctm_file) as f:
            for line in f:
                reco_id, channel, start, duration, symbol = line.strip().split()
                ctm_words.append((reco_id, int(channel), float(start), float(duration), symbol))
        ctm_words = sorted(ctm_words, key=lambda x:(x[0], x[2]))
        reco_to_ctm = defaultdict(list, {k: list(v) for k,v in groupby(ctm_words, key=lambda x:x[0])})
        segments = []
        num_total = len(ctm_words)
        num_overspanned = 0
        for reco_id in set([s.recording_id for s in self]):
            if reco_id in reco_to_ctm:
                for seg in self.find(recording_id=reco_id):
                    alignment = [AlignmentItem(symbol=word[4], start=word[2], duration=word[3]) for word in reco_to_ctm[reco_id] 
                                    if overspans(seg, TimeSpan(word[2], word[2] + word[3]))
                                    and (seg.channel == word[1] or not match_channel)
                                ]
                    num_overspanned += len(alignment)
                    segments.append(fastcopy(seg, alignment={type: alignment}))
            else:
                segments.append([s for s in self.find(recording_id=reco_id)])
        logging.info(f"{num_overspanned} alignments added out of {num_total} total. If there are several"
            " missing, there could be a mismatch problem.")
        return SupervisionSet.from_segments(segments)
                          
    def write_alignment_to_ctm(self, ctm_file: Pathlike, type: str = 'word') -> None:
        """
        Write alignments to CTM file.
        
        :param ctm_file: Path to output CTM file (will be created if not exists)
        :param type: Alignment type to write (default = `word`)
        """
        with open(ctm_file, 'w') as f:
            for s in self:
                if type in s.alignment:
                    for ali in s.alignment[type]:
                        f.write(f'{s.recording_id} {s.channel} {ali.start:.02f} {ali.duration:.02f} {ali.symbol}\n')
                
    def to_dicts(self) -> Iterable[dict]:
        return (s.to_dict() for s in self)

    def shuffle(self, rng: Optional[random.Random] = None) -> 'SupervisionSet':
        """
        Shuffle the supervision IDs in the current :class:`.SupervisionSet` and return a shuffled copy of self.

        :param rng: an optional instance of ``random.Random`` for precise control of randomness.
        :return: a shuffled copy of self.
        """
        if rng is None:
            rng = random
        ids = list(self.ids)
        rng.shuffle(ids)
        return SupervisionSet(segments={sid: self[sid] for sid in ids})

    def split(self, num_splits: int, shuffle: bool = False, drop_last: bool = False) -> List['SupervisionSet']:
        """
        Split the :class:`~lhotse.SupervisionSet` into ``num_splits`` pieces of equal size.

        :param num_splits: Requested number of splits.
        :param shuffle: Optionally shuffle the recordings order first.
        :param drop_last: determines how to handle splitting when ``len(seq)`` is not divisible
            by ``num_splits``. When ``False`` (default), the splits might have unequal lengths.
            When ``True``, it may discard the last element in some splits to ensure they are
            equally long.
        :return: A list of :class:`~lhotse.SupervisionSet` pieces.
        """
        return [
            SupervisionSet.from_segments(subset) for subset in
            split_sequence(self, num_splits=num_splits, shuffle=shuffle, drop_last=drop_last)
        ]

    def subset(self, first: Optional[int] = None, last: Optional[int] = None) -> 'SupervisionSet':
        """
        Return a new ``SupervisionSet`` according to the selected subset criterion.
        Only a single argument to ``subset`` is supported at this time.

        :param first: int, the number of first supervisions to keep.
        :param last: int, the number of last supervisions to keep.
        :return: a new ``SupervisionSet`` with the subset results.
        """
        assert exactly_one_not_null(first, last), "subset() can handle only one non-None arg."

        if first is not None:
            assert first > 0
            if first > len(self):
                logging.warning(f'SupervisionSet has only {len(self)} items but first {first} required; '
                                f'not doing anything.')
                return self
            return SupervisionSet.from_segments(islice(self, first))

        if last is not None:
            assert last > 0
            if last > len(self):
                logging.warning(f'SupervisionSet has only {len(self)} items but last {last} required; '
                                f'not doing anything.')
                return self
            return SupervisionSet.from_segments(islice(self, len(self) - last, len(self)))

    def filter(self, predicate: Callable[[SupervisionSegment], bool]) -> 'SupervisionSet':
        """
        Return a new SupervisionSet with the SupervisionSegments that satisfy the `predicate`.

        :param predicate: a function that takes a supervision as an argument and returns bool.
        :return: a filtered SupervisionSet.
        """
        return SupervisionSet.from_segments(seg for seg in self if predicate(seg))

    def map(self, transform_fn: Callable[[SupervisionSegment], SupervisionSegment]) -> 'SupervisionSet':
        """
        Map a ``transform_fn`` to the SupervisionSegments and return a new ``SupervisionSet``.

        :param transform_fn: a function that modifies a supervision as an argument.
        :return: a new ``SupervisionSet`` with modified segments.
        """
        return SupervisionSet.from_segments(s.map(transform_fn) for s in self)

    def transform_text(self, transform_fn: Callable[[str], str]) -> 'SupervisionSet':
        """
        Return a copy of the current ``SupervisionSet`` with the segments having a transformed ``text`` field.
        Useful for text normalization, phonetic transcription, etc.

        :param transform_fn: a function that accepts a string and returns a string.
        :return: a ``SupervisionSet`` with adjusted text.
        """
        return SupervisionSet.from_segments(s.transform_text(transform_fn) for s in self)

    def transform_alignment(self, transform_fn: Callable[[str], str], type: str = 'word') -> 'SupervisionSet':
        """
        Return a copy of the current ``SupervisionSet`` with the segments having a transformed ``alignment`` field.
        Useful for text normalization, phonetic transcription, etc.

        :param transform_fn: a function that accepts a string and returns a string.
        :param type:  alignment type to transform (key for alignment dict).
        :return: a ``SupervisionSet`` with adjusted text.
        """
        return SupervisionSet.from_segments(s.transform_alignment(transform_fn, type=type) for s in self)

    def find(
            self,
            recording_id: str,
            channel: Optional[int] = None,
            start_after: Seconds = 0,
            end_before: Optional[Seconds] = None,
            adjust_offset: bool = False,
            tolerance: Seconds = 0.001
    ) -> Iterable[SupervisionSegment]:
        """
        Return an iterable of segments that match the provided ``recording_id``.

        :param recording_id: Desired recording ID.
        :param channel: When specified, return supervisions in that channel - otherwise, in all channels.
        :param start_after: When specified, return segments that start after the given value.
        :param end_before: When specified, return segments that end before the given value.
        :param adjust_offset: When true, return segments as if the recordings had started at ``start_after``.
            This is useful for creating Cuts. Fom a user perspective, when dealing with a Cut, it is no
            longer helpful to know when the supervisions starts in a recording - instead, it's useful to
            know when the supervision starts relative to the start of the Cut.
            In the anticipated use-case, ``start_after`` and ``end_before`` would be
            the beginning and end of a cut;
            this option converts the times to be relative to the start of the cut.
        :param tolerance: Additional margin to account for floating point rounding errors
            when comparing segment boundaries.
        :return: An iterator over supervision segments satisfying all criteria.
        """
        segment_by_recording_id = self._index_by_recording_id_and_cache()
        return (
            # We only modify the offset - the duration remains the same, as we're only shifting the segment
            # relative to the Cut's start, and not truncating anything.
            segment.with_offset(-start_after) if adjust_offset else segment
            for segment in segment_by_recording_id.get(recording_id, [])
            if (channel is None or segment.channel == channel)
               and segment.start >= start_after - tolerance
               and (end_before is None or segment.end <= end_before + tolerance)
        )

    # This is a cache that significantly speeds up repeated ``find()`` queries.
    _segments_by_recording_id: Optional[Dict[str, List[SupervisionSegment]]] = None

    def _index_by_recording_id_and_cache(self):
        if self._segments_by_recording_id is None:
            from cytoolz import groupby
            self._segments_by_recording_id = groupby(lambda seg: seg.recording_id, self)
        return self._segments_by_recording_id

    def __repr__(self) -> str:
        return f'SupervisionSet(len={len(self)})'

    def __getitem__(self, sup_id_or_index: Union[int, str]) -> SupervisionSegment:
        if isinstance(sup_id_or_index, str):
            return self.segments[sup_id_or_index]
        # ~100x faster than list(dict.values())[index] for 100k elements
        return next(val for idx, val in enumerate(self.segments.values()) if idx == sup_id_or_index)

    def __contains__(self, item: Union[str, SupervisionSegment]) -> bool:
        if isinstance(item, str):
            return item in self.segments
        else:
            return item.id in self.segments

    def __iter__(self) -> Iterable[SupervisionSegment]:
        return iter(self.segments.values())

    def __len__(self) -> int:
        return len(self.segments)

    def __add__(self, other: 'SupervisionSet') -> 'SupervisionSet':
        return SupervisionSet(segments={**self.segments, **other.segments})
