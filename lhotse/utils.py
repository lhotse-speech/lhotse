from dataclasses import dataclass, asdict
from math import ceil, isclose
from pathlib import Path
from random import random
from typing import Union, Any, Dict

import numpy as np
import torch

Pathlike = Union[Path, str]

Seconds = float
Decibels = float

INT16MAX = 32768


def fix_random_seed(random_seed: int):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.random.manual_seed(random_seed)


def asdict_nonull(dclass) -> Dict[str, Any]:
    """Recursively convert a dataclass into a dict, removing the fields with `None` value on the top level."""
    return {k: v for k, v in asdict(dclass).items() if v is not None}


class SetContainingAnything:
    def __contains__(self, item):
        return True

    def intersection(self, iterable):
        return True


@dataclass
class TimeSpan:
    """Helper class for specifying a time span."""
    start: Seconds
    end: Seconds


# TODO: Ugh, Protocols are only in Python 3.8+...
def overlaps(lhs: Any, rhs: Any) -> bool:
    """Indicates whether two time-spans/segments are overlapping or not."""
    return lhs.start < rhs.end and rhs.start < lhs.end


def overspans(spanning: Any, spanned: Any) -> bool:
    """Indicates whether the left-hand-side time-span/segment covers the whole right-hand-side time-span/segment."""
    return spanning.start <= spanned.start <= spanned.end <= spanning.end


def time_diff_to_num_frames(time_diff: Seconds, frame_length: Seconds, frame_shift: Seconds) -> int:
    """Convert duration to an equivalent number of frames, so as to not exceed the duration."""
    if isclose(time_diff, 0.0):
        return 0
    return int(ceil((time_diff - frame_length) / frame_shift))
