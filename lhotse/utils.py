import gzip
import json
import math
import random
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from math import ceil, isclose
from pathlib import Path
from typing import Any, Dict, Union, Optional, Callable, List, TypeVar

import numpy as np
import torch
import yaml

Pathlike = Union[Path, str]

Seconds = float
Decibels = float

INT16MAX = 32768
LOG_EPSILON = -100.0
EPSILON = math.exp(LOG_EPSILON)

# This is a utility that generates uuid4's and is set when the user calls
# the ``fix_random_seed`` function.
# Python's uuid module is not affected by the ``random.seed(value)`` call,
# so we work around it to provide deterministic ID generation when requested.
_lhotse_uuid: Optional[Callable] = None


def fix_random_seed(random_seed: int):
    """
    Set the same random seed for the libraries and modules that Lhotse interacts with.
    Includes the ``random`` module, numpy, torch, and ``uuid4()`` function defined in this file.
    """
    global _lhotse_uuid
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.random.manual_seed(random_seed)
    # Ensure deterministic ID creation
    rd = random.Random()
    rd.seed(random_seed)
    _lhotse_uuid = lambda: uuid.UUID(int=rd.getrandbits(128))


def uuid4():
    """
    Generates uuid4's exactly like Python's uuid.uuid4() function.
    When ``fix_random_seed()`` is called, it will instead generate deterministic IDs.
    """
    if _lhotse_uuid is not None:
        return _lhotse_uuid()
    return uuid.uuid4()


def save_to_yaml(data: Any, path: Pathlike):
    compressed = str(path).endswith('.gz')
    opener = gzip.open if compressed else open
    mode = 'wt' if compressed else 'w'
    with opener(path, mode) as f:
        try:
            # When pyyaml is installed with C extensions, it can speed up the (de)serialization noticeably
            return yaml.dump(data, stream=f, Dumper=yaml.CSafeDumper)
        except AttributeError:
            return yaml.dump(data, stream=f, Dumper=yaml.SafeDumper)


def load_yaml(path: Pathlike) -> dict:
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path) as f:
        try:
            # When pyyaml is installed with C extensions, it can speed up the (de)serialization noticeably
            return yaml.load(stream=f, Loader=yaml.CSafeLoader)
        except AttributeError:
            return yaml.load(stream=f, Loader=yaml.SafeLoader)


class YamlMixin:
    def to_yaml(self, path: Pathlike):
        save_to_yaml(self.to_dicts(), path)

    @classmethod
    def from_yaml(cls, path: Pathlike):
        data = load_yaml(path)
        return cls.from_dicts(data)


def save_to_json(data: Any, path: Pathlike):
    """Save the data to a JSON file. Will use GZip to compress it if the path ends with a ``.gz`` extension."""
    compressed = str(path).endswith('.gz')
    opener = gzip.open if compressed else open
    mode = 'wt' if compressed else 'w'
    with opener(path, mode) as f:
        return json.dump(data, f, indent=2)


def load_json(path: Pathlike) -> Union[dict, list]:
    """Load a JSON file. Also supports compressed JSON with a ``.gz`` extension."""
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path) as f:
        return json.load(f)


class JsonMixin:
    def to_json(self, path: Pathlike):
        save_to_json(self.to_dicts(), path)

    @classmethod
    def from_json(cls, path: Pathlike):
        data = load_json(path)
        return cls.from_dicts(data)


def asdict_nonull(dclass) -> Dict[str, Any]:
    """
    Recursively convert a dataclass into a dict, removing all the fields with `None` value.
    Intended to use in place of dataclasses.asdict(), when the null values are not desired in the serialized document.
    """

    def non_null_dict_factory(collection):
        d = dict(collection)
        remove_keys = []
        for key, val in d.items():
            if val is None:
                remove_keys.append(key)
        for k in remove_keys:
            del d[k]
        return d

    return asdict(dclass, dict_factory=non_null_dict_factory)


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
    return lhs.start < rhs.end and rhs.start < lhs.end \
        and not isclose(lhs.start, rhs.end) and not isclose(rhs.start, lhs.end)


def overspans(spanning: Any, spanned: Any) -> bool:
    """Indicates whether the left-hand-side time-span/segment covers the whole right-hand-side time-span/segment."""
    return spanning.start <= spanned.start <= spanned.end <= spanning.end


def time_diff_to_num_frames(time_diff: Seconds, frame_length: Seconds, frame_shift: Seconds) -> int:
    """Convert duration to an equivalent number of frames, so as to not exceed the duration."""
    if isclose(time_diff, 0.0):
        return 0
    return int(ceil((time_diff - frame_length) / frame_shift))


def check_and_rglob(path: Pathlike, pattern: str) -> List[Path]:
    """
    Asserts that ``path`` exists, is a directory and contains at least one file satisfying the ``pattern``.

    :returns: a list of paths to files matching the ``pattern``.
    """
    path = Path(path)
    assert path.is_dir(), f'No such directory: {path}'
    matches = sorted(path.rglob(pattern))
    assert len(matches) > 0, f'No files matching pattern "{pattern}" in directory: {path}'
    return matches


@contextmanager
def recursion_limit(stack_size: int):
    """
    Code executed in this context will be able to recurse up to the specified recursion limit
    (or will hit a StackOverflow error if that number is too high).

    Usage:
        >>> with recursion_limit(1000):
        >>>     pass
    """
    import sys
    old_size = sys.getrecursionlimit()
    sys.setrecursionlimit(stack_size)
    try:
        yield
    finally:
        sys.setrecursionlimit(old_size)


T = TypeVar('T')


def fastcopy(dataclass_obj: T, **kwargs) -> T:
    """
    Returns a new object with the same member values.
    Selected members can be overwritten with kwargs.
    It's supposed to work only with dataclasses.
    It's 10X faster than the other methods I've tried...

    Example:
        >>> ts1 = TimeSpan(start=5, end=10)
        >>> ts2 = fastcopy(ts1, end=12)
    """
    return type(dataclass_obj)(**{**dataclass_obj.__dict__, **kwargs})
