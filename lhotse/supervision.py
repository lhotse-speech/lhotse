from dataclasses import asdict, dataclass
from typing import Callable, Dict, Iterable, Optional

from lhotse.utils import Pathlike, Seconds, asdict_nonull, load_yaml, save_to_yaml


@dataclass
class SupervisionSegment:
    id: str
    recording_id: str
    start: Seconds
    duration: Seconds
    channel_id: int = 0
    text: Optional[str] = None
    language: Optional[str] = None
    speaker: Optional[str] = None
    gender: Optional[str] = None

    @property
    def end(self) -> Seconds:
        # Precision up to 1ms to avoid float numeric artifacts
        return round(self.start + self.duration, ndigits=3)

    def with_offset(self, offset: Seconds) -> 'SupervisionSegment':
        kwargs = asdict(self)
        # Precision up to 1ms to avoid float numeric artifacts
        kwargs['start'] = round(kwargs['start'] + offset, ndigits=3)
        return SupervisionSegment(**kwargs)

    @staticmethod
    def from_dict(data: dict) -> 'SupervisionSegment':
        return SupervisionSegment(**data)


@dataclass
class SupervisionSet:
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
        return SupervisionSet(segments={s.id: s for s in segments})

    @staticmethod
    def from_yaml(path: Pathlike) -> 'SupervisionSet':
        raw_segments = load_yaml(path)
        return SupervisionSet.from_segments(SupervisionSegment.from_dict(s) for s in raw_segments)

    def to_yaml(self, path: Pathlike):
        data = [asdict_nonull(s) for s in self]
        save_to_yaml(data, path)

    def filter(self, predicate: Callable[[SupervisionSegment], bool]) -> 'SupervisionSet':
        """
        Return a new SupervisionSet with the SupervisionSegments that satisfy the `predicate`.

        :param predicate: a function that takes a supervision as an argument and returns bool.
        :return: a filtered SupervisionSet.
        """
        return SupervisionSet.from_segments(seg for seg in self if predicate(seg))

    def find(
            self,
            recording_id: str,
            channel: int,
            start_after: Seconds = 0,
            end_before: Optional[Seconds] = None,
            adjust_offset: bool = False
    ) -> Iterable[SupervisionSegment]:
        """
        Return an iterable of segments that match the provided ``recording_id``.

        :param recording_id: Desired recording ID.
        :param channel: Return supervisions in the specified channel.
        :param start_after: When specified, return segments that start after the given value.
        :param end_before: When specified, return segments that end before the given value.
        :param adjust_offset: When true, return segments as if the recordings had started at ``start_after``.
            In other words, :math:`segment.start = segment.start - start_after`.
        :return: An iterator over supervision segments satisfying all criteria.
        """
        return (
            segment.with_offset(-start_after) if adjust_offset else segment
            for segment in self
            if segment.recording_id == recording_id
               and segment.channel_id == channel
               and segment.start >= start_after
               and (end_before is None or segment.end <= end_before)
        )

    def __getitem__(self, item: str) -> SupervisionSegment:
        return self.segments[item]

    def __iter__(self) -> Iterable[SupervisionSegment]:
        return iter(self.segments.values())

    def __len__(self) -> int:
        return len(self.segments)

    def __add__(self, other: 'SupervisionSet') -> 'SupervisionSet':
        return SupervisionSet(segments={**self.segments, **other.segments})
