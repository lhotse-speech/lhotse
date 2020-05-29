from dataclasses import dataclass
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
