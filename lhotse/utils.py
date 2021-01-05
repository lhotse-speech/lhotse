import gzip
import json
import math
import random
import uuid
from contextlib import AbstractContextManager, contextmanager
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_HALF_DOWN, ROUND_HALF_UP
from math import ceil, isclose
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, TypeVar, Union

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

Pathlike = Union[Path, str]
T = TypeVar('T')

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


def split_sequence(seq: Sequence[Any], num_splits: int, shuffle: bool = False) -> List[List[Any]]:
    """
    Split a sequence into ``num_splits`` equal parts. The element order can be randomized.
    Raises a ``ValueError`` if ``num_splits`` is larger than ``len(seq)``.

    :param seq: an input iterable (can be a Lhotse manifest).
    :param num_splits: how many output splits should be created.
    :param shuffle: optionally shuffle the sequence before splitting.
    :return: a list of length ``num_splits`` containing smaller lists (the splits).
    """
    seq = list(seq)
    num_items = len(seq)
    if num_splits > num_items:
        raise ValueError(f"Cannot split iterable into more chunks ({num_splits}) than its number of items {num_items}")
    if shuffle:
        random.shuffle(seq)
    chunk_size = int(ceil(num_items / num_splits))
    split_indices = [(i * chunk_size, min(num_items, (i + 1) * chunk_size)) for i in range(num_splits)]
    return [seq[begin: end] for begin, end in split_indices]


def compute_num_frames(
        duration: Seconds,
        frame_shift: Seconds,
        rounding: str = ROUND_HALF_UP
) -> int:
    """
    Compute the number of frames from duration and frame_shift in a safe way.

    The need for this function is best illustrated with an example edge case:

    >>> duration = 16.165
    >>> frame_shift = 0.01
    >>> num_frames = duration / frame_shift  # we expect it to finally be 1617
    >>> num_frames
      1616.4999999999998
    >>> round(num_frames)
      1616
    >>> Decimal(num_frames).quantize(0, rounding=ROUND_HALF_UP)
      1616
    >>> round(num_frames, ndigits=8)
      1616.5
    >>> Decimal(round(num_frames, ndigits=8)).quantize(0, rounding=ROUND_HALF_UP)
      1617
    """
    return int(
        Decimal(
            # 8 is a good number because cases like 14.49175 still work correctly,
            # while problematic cases like 14.49999999998 are typically breaking much later than 8th decimal
            # with double-precision floats.
            round(duration / frame_shift, ndigits=8)
        ).quantize(0, rounding=rounding)
    )


def during_docs_build() -> bool:
    import os
    return bool(os.environ.get('READTHEDOCS'))


def tqdm_urlretrieve_hook(t):
    """Wraps tqdm instance.
    Don't forget to close() or __exit__()
    the tqdm instance once you're done with it (easiest using `with` syntax).
    Example
    -------
    >>> from urllib.request import urlretrieve
    >>> with tqdm(...) as t:
    ...     reporthook = tqdm_urlretrieve_hook(t)
    ...     urlretrieve(..., reporthook=reporthook)

    Source: https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py
    """
    last_b = [0]

    def update_to(b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] or -1,
            remains unchanged.
        """
        if tsize not in (None, -1):
            t.total = tsize
        displayed = t.update((b - last_b[0]) * bsize)
        last_b[0] = b
        return displayed

    return update_to


def urlretrieve_progress(url, filename=None, data=None, desc=None):
    """
    Works exactly like urllib.request.urlretrieve, but attaches a tqdm hook to display
    a progress bar of the download.
    Use "desc" argument to display a user-readable string that informs what is being downloaded.
    """
    from urllib.request import urlretrieve
    with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=desc) as t:
        reporthook = tqdm_urlretrieve_hook(t)
        return urlretrieve(url=url, filename=filename, reporthook=reporthook, data=data)


class nullcontext(AbstractContextManager):
    """Context manager that does no additional processing.

    Used as a stand-in for a normal context manager, when a particular
    block of code is only sometimes used with a normal context manager:

    cm = optional_cm if condition else nullcontext()
    with cm:
        # Perform operation, using optional_cm if condition is True

    Note(pzelasko): This is copied from Python 3.7 stdlib so that we can use it in 3.6.
    """

    def __init__(self, enter_result=None):
        self.enter_result = enter_result

    def __enter__(self):
        return self.enter_result

    def __exit__(self, *excinfo):
        pass


def perturb_num_samples(num_samples: int, factor: float) -> int:
    """Mimicks the behavior of the speed perturbation on the number of samples."""
    rounding = ROUND_HALF_UP if factor >= 1.0 else ROUND_HALF_DOWN
    return int(Decimal(round(num_samples / factor, ndigits=8)).quantize(0, rounding=rounding))


def compute_num_samples(duration: Seconds, sampling_rate: int, rounding=ROUND_HALF_UP) -> int:
    """
    Convert a time quantity to the number of samples given a specific sampling rate.
    Performs consistent rounding up or down for ``duration`` that is not a multiply of
    the sampling interval (unlike Python's built-in ``round()`` that implements banker's rounding).
    """
    return int(
        Decimal(
            duration * sampling_rate
        ).quantize(0, rounding=rounding)
    )


def index_by_id_and_check(manifests: Iterable[T]) -> Dict[str, T]:
    id2man = {}
    for m in manifests:
        assert m.id not in id2man, f"Duplicated manifest ID: {m.id}"
        id2man[m.id] = m
    return id2man
