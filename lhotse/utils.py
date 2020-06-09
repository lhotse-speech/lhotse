from dataclasses import dataclass, asdict
from math import ceil
from pathlib import Path
from random import random
from typing import Union, Any, Dict

import numpy as np
import torch

Pathlike = Union[Path, str]

Seconds = float
Milliseconds = float
Decibels = float

INT16MAX = 32768


def fix_random_seed(random_seed: int):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.random.manual_seed(random_seed)


def asdict_nonull(dclass) -> Dict[str, Any]:
    return {k: v for k, v in asdict(dclass).items() if v is not None}


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
