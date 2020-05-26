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
    begin: Seconds
    end: Seconds


# TODO: Ugh, Protocols are only in Python 3.8+...
def overlaps(lhs: Any, rhs: Any) -> bool:
    return (
            lhs.begin < rhs.end
            and rhs.begin < lhs.end
    )


def overspans(spanning: Any, spanned: Any) -> bool:
    return spanning.begin <= spanned.begin <= spanned.end <= spanning.end
