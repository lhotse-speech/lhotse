from dataclasses import dataclass, asdict
from typing import Dict, Optional, Iterable

from lhotse.utils import Seconds, Pathlike, asdict_nonull, load_yaml, save_to_yaml


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
        return self.start + self.duration

    def with_offset(self, offset: Seconds) -> 'SupervisionSegment':
        kwargs = asdict(self)
        kwargs['start'] += offset
        return SupervisionSegment(**kwargs)


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
    def from_yaml(path: Pathlike) -> 'SupervisionSet':
        raw_segments = load_yaml(path)
        return SupervisionSet(segments={s['id']: SupervisionSegment(**s) for s in raw_segments})

    def to_yaml(self, path: Pathlike):
        data = [asdict_nonull(s) for s in self]
        save_to_yaml(data, path)

    def get(self, segment_id: str) -> SupervisionSegment:
        return self.segments[segment_id]

    def __iter__(self) -> Iterable[SupervisionSegment]:
        return iter(self.segments.values())

    def __len__(self) -> int:
        return len(self.segments)

    def __add__(self, other: 'SupervisionSet') -> 'SupervisionSet':
        return SupervisionSet(segments={**self.segments, **other.segments})
