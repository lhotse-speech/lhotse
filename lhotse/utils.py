from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Union, Any

Pathlike = Union[Path, str]

Seconds = float
Milliseconds = float
Decibels = float

INT16MAX = 32768


class SetContainingAnything:
    def __contains__(self, item):
        return True

    def intersection(self, iterable):
        return True


@dataclass
class TimeSpan:
    start: Seconds
    end: Seconds


# TODO: Ugh, Protocols are only in Python 3.8+...
def overlaps(lhs: Any, rhs: Any) -> bool:
    return (
            lhs.start < rhs.end
            and rhs.start < lhs.end
    )


def overspans(spanning: Any, spanned: Any) -> bool:
    return spanning.start <= spanned.start <= spanned.end <= spanning.end


def time_diff_to_num_frames(time_diff: Seconds, frame_length: Seconds, frame_shift: Seconds) -> int:
    """Convert duration to an equivalent number of frames, so as to not exceed the duration."""
    return int(ceil((time_diff - frame_length) / frame_shift))
