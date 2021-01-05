from dataclasses import dataclass
from math import floor

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from lhotse.utils import JsonMixin, Seconds, YamlMixin, asdict_nonull, fastcopy, index_by_id_and_check, \
    perturb_num_samples, split_sequence


@dataclass(frozen=True, unsafe_hash=True)
class SupervisionSegment:
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

    @property
    def end(self) -> Seconds:
        return round(self.start + self.duration, ndigits=8)

    def with_offset(self, offset: Seconds) -> 'SupervisionSegment':
        """Return an identical ``SupervisionSegment``, but with the ``offset`` added to the ``start`` field."""
        return fastcopy(self, start=round(self.start + offset, ndigits=8))

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
        start_sample = round(self.start * sampling_rate)
        num_samples = round(self.duration * sampling_rate)
        new_start = perturb_num_samples(start_sample, factor) / sampling_rate
        new_duration = perturb_num_samples(num_samples, factor) / sampling_rate
        return fastcopy(
            self,
            id=f'{self.id}_sp{factor}' if affix_id else self.id,
            recording_id=f'{self.recording_id}_sp{factor}' if affix_id else self.id,
            start=new_start,
            duration=new_duration
        )

    def trim(self, end: Seconds) -> 'SupervisionSegment':
        """
        Return an identical ``SupervisionSegment``, but ensure that ``self.start`` is not negative (in which case
        it's set to 0) and ``self.end`` does not exceed the ``end`` parameter.

        This method is useful for ensuring that the supervision does not exceed a cut's bounds,
        in which case pass ``cut.duration`` as the ``end`` argument, since supervision times are relative to the cut.
        """
        start_exceeds_by = abs(min(0, self.start))
        end_exceeds_by = max(0, self.end - end)
        return fastcopy(self, start=max(0, self.start), duration=self.duration - end_exceeds_by - start_exceeds_by)

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

    @staticmethod
    def from_dict(data: dict) -> 'SupervisionSegment':
        return SupervisionSegment(**data)


@dataclass
class SupervisionSet(JsonMixin, YamlMixin, Sequence[SupervisionSegment]):
    """
    SupervisionSet represents a collection of segments containing some supervision information.
    The only required fields are the ID of the segment, ID of the corresponding recording,
    and the start and duration of the segment in seconds.
    All other fields, such as text, language or speaker, are deliberately optional
    to support a wide range of tasks, as well as adding more supervision types in the future,
    while retaining backwards compatibility.
    """
    segments: Dict[str, SupervisionSegment]

    @staticmethod
    def from_segments(segments: Iterable[SupervisionSegment]) -> 'SupervisionSet':
        return SupervisionSet(segments=index_by_id_and_check(segments))

    @staticmethod
    def from_dicts(data: Iterable[Dict]) -> 'SupervisionSet':
        return SupervisionSet.from_segments(SupervisionSegment.from_dict(s) for s in data)

    def to_dicts(self) -> List[dict]:
        return [asdict_nonull(s) for s in self]

    def split(self, num_splits: int, shuffle: bool = False) -> List['SupervisionSet']:
        """
        Split the ``SupervisionSet`` into ``num_splits`` pieces of equal size.

        :param num_splits: Requested number of splits.
        :param shuffle: Optionally shuffle the supervisions order first.
        :return: A list of ``SupervisionSet`` pieces.
        """
        return [
            SupervisionSet.from_segments(subset) for subset in
            split_sequence(self, num_splits=num_splits, shuffle=shuffle)
        ]

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

    def find(
            self,
            recording_id: str,
            channel: Optional[int] = None,
            start_after: Seconds = 0,
            end_before: Optional[Seconds] = None,
            adjust_offset: bool = False
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
        :return: An iterator over supervision segments satisfying all criteria.
        """
        segment_by_recording_id = self._index_by_recording_id_and_cache()
        return (
            # We only modify the offset - the duration remains the same, as we're only shifting the segment
            # relative to the Cut's start, and not truncating anything.
            segment.with_offset(-start_after) if adjust_offset else segment
            for segment in segment_by_recording_id[recording_id]
            if (channel is None or segment.channel == channel)
               and segment.start >= start_after
               and (end_before is None or segment.end <= end_before)
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

    def __getitem__(self, item: str) -> SupervisionSegment:
        return self.segments[item]

    def __iter__(self) -> Iterable[SupervisionSegment]:
        return iter(self.segments.values())

    def __len__(self) -> int:
        return len(self.segments)

    def __add__(self, other: 'SupervisionSet') -> 'SupervisionSet':
        return SupervisionSet(segments={**self.segments, **other.segments})
