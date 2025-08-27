import ast
import functools
import hashlib
import inspect
import logging
import math
import os
import random
import secrets
import sys
import urllib
import uuid
import warnings
from contextlib import AbstractContextManager, contextmanager
from dataclasses import asdict, dataclass
from decimal import ROUND_HALF_DOWN, ROUND_HALF_UP, Decimal
from itertools import chain
from math import ceil, isclose
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)
from urllib.parse import urlparse, urlunparse

import click
import numpy as np
import torch
from tqdm.auto import tqdm

Pathlike = Union[Path, str]
T = TypeVar("T")

Seconds = float
Decibels = float

INT16MAX = 32768
EPSILON = 1e-10
LOG_EPSILON = math.log(EPSILON)
DEFAULT_PADDING_VALUE = 0  # used for custom attrs

# This is a utility that generates uuid4's and is set when the user calls
# the ``fix_random_seed`` function.
# Python's uuid module is not affected by the ``random.seed(value)`` call,
# so we work around it to provide deterministic ID generation when requested.
_lhotse_uuid: Optional[Callable] = None


class SmartOpen:
    """Wrapper class around smart_open.open method

    The smart_open.open attributes are cached as classed attributes - they play the role of singleton pattern.

    The SmartOpen.setup method is intended for initial setup.
    It imports the `open` method from the optional `smart_open` Python package,
    and sets the parameters which are shared between all calls of the `smart_open.open` method.

    If you do not call the setup method it is called automatically in SmartOpen.open with the provided parameters.

    The example demonstrates that instantiating S3 `session.client` once,
    instead using the defaults and leaving the smart_open creating it every time
    has dramatic performance benefits.

    Example::

        >>> import boto3
        >>> session = boto3.Session()
        >>> client = session.client('s3')
        >>> from lhotse.utils import SmartOpen
        >>>
        >>> if not slow:
        >>>     # Reusing a single client speeds up the smart_open.open calls
        >>>     SmartOpen.setup(transport_params=dict(client=client))
        >>>
        >>> # Simulating SmartOpen usage as in Lhotse data structures: AudioSource, Features, etc.
        >>> for i in range(1000):
        >>>     SmartOpen.open(s3_url, 'rb') as f:
        >>>         source = f.read()
    """

    transport_params: Optional[Dict] = None
    import_err_msg = (
        "Please do 'pip install smart_open' - "
        "if you are using S3/GCP/Azure/other cloud-specific URIs, do "
        "'pip install smart_open[s3]' (or smart_open[gcp], etc.) instead."
    )
    smart_open: Optional[Callable] = None

    @classmethod
    def setup(cls, transport_params: Optional[dict] = None):
        try:
            from smart_open import open as sm_open
        except ImportError:
            raise ImportError(cls.import_err_msg)
        if (
            cls.transport_params is not None
            and cls.transport_params != transport_params
        ):
            logging.warning(
                f"SmartOpen.setup second call overwrites existing transport_params with new version"
                f"\t\n{cls.transport_params}\t\nvs\t\n{transport_params}"
            )
        cls.transport_params = transport_params
        cls.smart_open = sm_open

    @classmethod
    def open(cls, uri, mode="rb", transport_params=None, **kwargs):
        if cls.smart_open is None:
            cls.setup(transport_params=transport_params)
        transport_params = (
            transport_params if transport_params else cls.transport_params
        )
        return cls.smart_open(
            uri,
            mode=mode,
            transport_params=transport_params,
            **kwargs,
        )


def is_valid_url(value: str) -> bool:
    try:
        result = urlparse(value)
        return bool(result.scheme) and bool(result.netloc)
    except AttributeError:
        return False


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

    @property
    def duration(self) -> Seconds:
        return self.end - self.start


# TODO: Ugh, Protocols are only in Python 3.8+...
def overlaps(lhs: Any, rhs: Any) -> bool:
    """Indicates whether two time-spans/segments are overlapping or not."""
    return (
        lhs.start < rhs.end
        and rhs.start < lhs.end
        and not isclose(lhs.start, rhs.end)
        and not isclose(rhs.start, lhs.end)
    )


def overspans(spanning: Any, spanned: Any, tolerance: float = 1e-3) -> bool:
    """Indicates whether the left-hand-side time-span/segment covers the whole right-hand-side time-span/segment."""
    # We add a small epsilon to the comparison to avoid floating-point precision issues.
    return (
        spanning.start - tolerance
        <= spanned.start
        <= spanned.end
        <= spanning.end + tolerance
    )


def time_diff_to_num_frames(
    time_diff: Seconds, frame_length: Seconds, frame_shift: Seconds
) -> int:
    """Convert duration to an equivalent number of frames, so as to not exceed the duration."""
    if isclose(time_diff, 0.0):
        return 0
    return int(ceil((time_diff - frame_length) / frame_shift))


def check_and_rglob(
    path: Pathlike, pattern: str, strict: Optional[bool] = True
) -> List[Path]:
    """
    Asserts that ``path`` exists, is a directory and contains at least one file satisfying the ``pattern``.
    If `strict` is False, then zero matches are allowed.

    :returns: a list of paths to files matching the ``pattern``.
    """
    path = Path(path)
    assert path.is_dir(), f"No such directory: {path}"
    matches = sorted(path.rglob(pattern))
    assert (
        len(matches) > 0 or not strict
    ), f'No files matching pattern "{pattern}" in directory: {path}'
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


def split_manifest_lazy(
    it: Iterable[Any],
    output_dir: Pathlike,
    chunk_size: int,
    prefix: str = "",
    num_digits: int = 8,
    start_idx: int = 0,
) -> List:
    """
    Splits a manifest (either lazily or eagerly opened) into chunks, each
    with ``chunk_size`` items (except for the last one, typically).

    In order to be memory efficient, this implementation saves each chunk
    to disk in a ``.jsonl.gz`` format as the input manifest is sampled.

    .. note:: For lowest memory usage, use ``load_manifest_lazy`` to open the
        input manifest for this method.

    :param it: any iterable of Lhotse manifests.
    :param output_dir: directory where the split manifests are saved.
        Each manifest is saved at: ``{output_dir}/{prefix}.{split_idx}.jsonl.gz``
    :param chunk_size: the number of items in each chunk.
    :param prefix: the prefix of each manifest.
    :param num_digits: the width of ``split_idx``, which will be left padded with zeros to achieve it.
    :param start_idx: The split index to start counting from (default is ``0``).
    :return: a list of lazily opened chunk manifests.
    """
    from lhotse.serialization import SequentialJsonlWriter

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if prefix == "":
        prefix = "split"

    split_idx = start_idx
    splits = []
    items = iter(it)
    try:
        item = next(items)
    except StopIteration:
        return splits
    while True:
        try:
            written = 0
            idx = f"{split_idx}".zfill(num_digits)
            with SequentialJsonlWriter(
                (output_dir / prefix).with_suffix(f".{idx}.jsonl.gz")
            ) as writer:
                while written < chunk_size:
                    writer.write(item)
                    written += 1
                    item = next(items)
            split_idx += 1
        except StopIteration:
            break
        finally:
            subcutset = writer.open_manifest()
            if subcutset is not None:
                # Edge case: there were exactly chunk_size cuts in the cutset
                splits.append(subcutset)

    return splits


def split_sequence(
    seq: Sequence[Any], num_splits: int, shuffle: bool = False, drop_last: bool = False
) -> List[List[Any]]:
    """
    Split a sequence into ``num_splits`` equal parts. The element order can be randomized.
    Raises a ``ValueError`` if ``num_splits`` is larger than ``len(seq)``.

    :param seq: an input iterable (can be a Lhotse manifest).
    :param num_splits: how many output splits should be created.
    :param shuffle: optionally shuffle the sequence before splitting.
    :param drop_last: determines how to handle splitting when ``len(seq)`` is not divisible
        by ``num_splits``. When ``False`` (default), the splits might have unequal lengths.
        When ``True``, it may discard the last element in some splits to ensure they are
        equally long.
    :return: a list of length ``num_splits`` containing smaller lists (the splits).
    """
    seq = list(seq)
    num_items = len(seq)
    if num_splits > num_items:
        raise ValueError(
            f"Cannot split iterable into more chunks ({num_splits}) than its number of items {num_items}"
        )
    if shuffle:
        random.shuffle(seq)
    chunk_size = num_items // num_splits

    num_shifts = num_items % num_splits
    if drop_last:
        # Equally-sized splits; discards the remainder by default, no shifts are needed
        end_shifts = [0] * num_splits
        begin_shifts = [0] * num_splits
    else:
        # Non-equally sized splits; need to shift the indices like:
        # [0, 10] -> [0, 11]    (begin_shift=0, end_shift=1)
        # [10, 20] -> [11, 22]  (begin_shift=1, end_shift=2)
        # [20, 30] -> [22, 32]  (begin_shift=2, end_shift=2)
        # for num_items=32 and num_splits=3
        end_shifts = list(range(1, num_shifts + 1)) + [num_shifts] * (
            num_splits - num_shifts
        )
        begin_shifts = [0] + end_shifts[:-1]

    split_indices = [
        [i * chunk_size + begin_shift, (i + 1) * chunk_size + end_shift]
        for i, begin_shift, end_shift in zip(
            range(num_splits), begin_shifts, end_shifts
        )
    ]
    splits = [seq[begin:end] for begin, end in split_indices]
    return splits


def compute_num_frames(
    duration: Seconds,
    frame_shift: Seconds,
    sampling_rate: int,
) -> int:
    """
    Compute the number of frames from duration and frame_shift in a safe way.
    """
    num_samples = round(duration * sampling_rate)
    window_hop = round(frame_shift * sampling_rate)
    num_frames = int((num_samples + window_hop // 2) // window_hop)
    return num_frames


def compute_num_frames_from_samples(
    num_samples: int,
    frame_shift: Seconds,
    sampling_rate: int,
) -> int:
    """
    Compute the number of frames from number of samples and frame_shift in a safe way.
    """
    window_hop = round(frame_shift * sampling_rate)
    num_frames = int((num_samples + window_hop // 2) // window_hop)
    return num_frames


def compute_num_windows(sig_len: Seconds, win_len: Seconds, hop: Seconds) -> int:
    """
    Return a number of windows obtained from signal of length equal to ``sig_len``
    with windows of ``win_len`` and ``hop`` denoting shift between windows.
    Examples:
    ```
      (sig_len,win_len,hop) -> num_windows # list of windows times
      (1, 6.1, 3) -> 1  # 0-1
      (3, 1, 6.1) -> 1  # 0-1
      (3, 6.1, 1) -> 1  # 0-3
      (5.9, 1, 3) -> 2  # 0-1, 3-4
      (5.9, 3, 1) -> 4  # 0-3, 1-4, 2-5, 3-5.9
      (6.1, 1, 3) -> 3  # 0-1, 3-4, 6-6.1
      (6.1, 3, 1) -> 5  # 0-3, 1-4, 2-5, 3-6, 4-6.1
      (5.9, 3, 3) -> 2  # 0-3, 3-5.9
      (6.1, 3, 3) -> 3  # 0-3, 3-6, 6-6.1
      (0.0, 3, 3) -> 0
    ```
    :param sig_len: Signal length in seconds.
    :param win_len: Window length in seconds
    :param hop: Shift between windows in seconds.
    :return: Number of windows in signal.
    """
    n = ceil(max(sig_len - win_len, 0) / hop)
    b = (sig_len - n * hop) > 0
    return (sig_len > 0) * (n + int(b))


def during_docs_build() -> bool:
    import os

    return bool(os.environ.get("READTHEDOCS"))


def resumable_download(
    url: str,
    filename: Pathlike,
    force_download: bool = False,
    completed_file_size: Optional[int] = None,
    missing_ok: bool = False,
) -> None:
    # Check if the file exists and get its size
    file_exists = os.path.exists(filename)
    if file_exists:
        if force_download:
            logging.info(
                f"Removing existing file and downloading from scratch because force_download=True: {filename}"
            )
            os.unlink(filename)
        file_size = os.path.getsize(filename)

        if completed_file_size and file_size == completed_file_size:
            return
    else:
        file_size = 0

    # Set the request headers to resume downloading
    # Also set user-agent header to stop picky servers from complaining with 403
    ua_headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_4) AppleWebKit/603.1.30 (KHTML, like Gecko) Version/10.1 Safari/603.1.30",
    }

    headers = {
        "Range": "bytes={}-".format(file_size),
        **ua_headers,
    }

    # Create a request object with the URL and headers
    req = urllib.request.Request(url, headers=headers)

    # Open the file for writing in binary mode and seek to the end
    # r+b is needed in order to allow seeking at the beginning of a file
    # when downloading from scratch
    mode = "r+b" if file_exists else "wb"
    with open(filename, mode) as f:

        def _download(rq, size):
            f.seek(size, 0)
            # just in case some garbage was written to the file, truncate it
            f.truncate()

            # Open the URL and read the contents in chunks
            with urllib.request.urlopen(rq) as response:
                chunk_size = 1024
                total_size = int(response.headers.get("content-length", 0)) + size
                with tqdm(
                    total=total_size,
                    initial=size,
                    unit="B",
                    unit_scale=True,
                    desc=str(filename),
                ) as pbar:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        pbar.update(len(chunk))

        try:
            _download(req, file_size)
        except urllib.error.HTTPError as e:
            # "Request Range Not Satisfiable" means the requested range
            # starts after the file ends OR that the server does not support range requests.
            if e.code == 404 and missing_ok:
                logging.warning(
                    f"{url} does not exist (error 404). Skipping this file."
                )
                if Path(filename).is_file():
                    os.remove(filename)
            elif e.code == 416:
                content_range = e.headers.get("Content-Range", None)
                if content_range is None:
                    # sometimes, the server actually supports range requests
                    # but does not return the Content-Range header with 416 code
                    # This is out of spec, but let us check twice for pragmatic reasons.
                    head_req = urllib.request.Request(url, method="HEAD")
                    head_res = urllib.request.urlopen(head_req)
                    if head_res.headers.get("Accept-Ranges", "none") != "none":
                        content_length = head_res.headers.get("Content-Length")
                        content_range = f"bytes */{content_length}"

                if content_range == f"bytes */{file_size}":
                    # If the content-range returned by server also matches the file size,
                    # then the file is already downloaded
                    logging.info(f"File already downloaded: {filename}")
                else:
                    logging.info(
                        "Server does not support range requests - attempting downloading from scratch"
                    )
                    _download(urllib.request.Request(url, headers=ua_headers), 0)
            else:
                raise e


def _is_within_directory(directory: Path, target: Path):

    abs_directory = directory.resolve()
    abs_target = target.resolve()

    return abs_directory in abs_target.parents


def safe_extract(
    tar: Any,
    path: Pathlike = ".",
    members: Optional[List[str]] = None,
    *,
    numeric_owner: bool = False,
) -> None:
    """
    Extracts a tar file in a safe way, avoiding path traversal attacks.
    See: https://github.com/lhotse-speech/lhotse/pull/872
    """

    path = Path(path)

    for member in tar.getmembers():
        member_path = path / member.name
        if not _is_within_directory(path, member_path):
            raise Exception("Attempted Path Traversal in Tar File")

    tar.extractall(path, members, numeric_owner=numeric_owner)


def safe_extract_rar(
    rar: Any,
    path: Pathlike = ".",
    members: Optional[List[str]] = None,
) -> None:
    """
    Extracts a rar file in a safe way, avoiding path traversal attacks.
    """

    path = Path(path)

    for member in rar.infolist():
        member_path = path / member.filename
        if not _is_within_directory(path, member_path):
            raise Exception("Attempted Path Traversal in Rar File")

    rar.extractall(path, members)


class nullcontext(AbstractContextManager):
    """Context manager that does no additional processing.

    Used as a stand-in for a normal context manager, when a particular
    block of code is only sometimes used with a normal context manager:

    cm = optional_cm if condition else nullcontext()
    with cm:
        # Perform operation, using optional_cm if condition is True

    Note(pzelasko): This is copied from Python 3.7 stdlib so that we can use it in 3.6.
    """

    def __init__(self, enter_result=None, *args, **kwargs):
        self.enter_result = enter_result

    def __enter__(self):
        return self.enter_result

    def __exit__(self, *excinfo):
        pass


def perturb_num_samples(num_samples: int, factor: float) -> int:
    """Mimicks the behavior of the speed perturbation on the number of samples."""
    rounding = ROUND_HALF_UP if factor >= 1.0 else ROUND_HALF_DOWN
    return int(
        Decimal(round(num_samples / factor, ndigits=8)).quantize(0, rounding=rounding)
    )


def compute_num_samples(
    duration: Seconds, sampling_rate: Union[int, float], rounding=ROUND_HALF_UP
) -> int:
    """
    Convert a time quantity to the number of samples given a specific sampling rate.
    Performs consistent rounding up or down for ``duration`` that is not a multiply of
    the sampling interval (unlike Python's built-in ``round()`` that implements banker's rounding).
    """
    return int(
        Decimal(round(duration * sampling_rate, ndigits=8)).quantize(
            0, rounding=rounding
        )
    )


def add_durations(*durs: Seconds, sampling_rate: int) -> Seconds:
    """
    Adds two durations in a way that avoids floating point precision issues.
    The durations in seconds are first converted to audio sample counts,
    then added, and finally converted back to floating point seconds.
    """
    tot_num_samples = sum(
        compute_num_samples(d, sampling_rate=sampling_rate) for d in durs
    )
    return tot_num_samples / sampling_rate


def compute_start_duration_for_extended_cut(
    start: Seconds,
    duration: Seconds,
    new_duration: Seconds,
    direction: Literal["center", "left", "right", "random"] = "center",
) -> Tuple[Seconds, Seconds]:
    """
    Compute the new value of "start" for a time interval characterized by ``start`` and ``duration``
    that is being extended to ``new_duration`` towards ``direction``.
    :return: a new value of ``start`` and ``new_duration`` -- adjusted for possible negative start.
    """

    if new_duration <= duration:
        # New duration is shorter; do nothing.
        return start, duration

    if direction == "center":
        new_start = start - (new_duration - duration) / 2
    elif direction == "left":
        new_start = start - (new_duration - duration)
    elif direction == "right":
        new_start = start
    elif direction == "random":
        new_start = random.uniform(start - (new_duration - duration), start)
    else:
        raise ValueError(f"Unexpected direction: {direction}")

    if new_start < 0:
        # We exceeded the start of the recording.
        # We'll decrease the new_duration by the negative offset.
        new_duration = round(new_duration + new_start, ndigits=15)
        new_start = 0

    return round(new_start, ndigits=15), new_duration


def merge_items_with_delimiter(
    values: Iterable[str],
    prefix: str = "cat",
    delimiter: str = "#",
    return_first: bool = False,
) -> Optional[str]:
    # e.g.
    # values = ["1125-76840-0001", "1125-53670-0003"]
    # return "cat#1125-76840-0001#1125-53670-0003"
    # if return_first is True, return "1125-76840-0001"
    values = list(values)
    if len(values) == 0:
        return None
    if len(values) == 1 or return_first:
        return values[0]
    return delimiter.join(chain([prefix], values))


def exactly_one_not_null(*args) -> bool:
    not_null = [arg is not None for arg in args]
    return sum(not_null) == 1


def supervision_to_frames(
    supervision,
    frame_shift: Seconds,
    sampling_rate: int,
    max_frames: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Utility to convert a supervision's time span into a tuple of ``(start_frame, num_frames)``.
    When ``max_frames`` is specified, it will truncate the ``num_frames`` (if necessary).
    """
    start_frame = compute_num_frames(
        supervision.start, frame_shift=frame_shift, sampling_rate=sampling_rate
    )
    num_frames = compute_num_frames(
        supervision.duration, frame_shift=frame_shift, sampling_rate=sampling_rate
    )
    if max_frames:
        diff = start_frame + num_frames - max_frames
        if diff > 0:
            num_frames -= diff
    return start_frame, num_frames


def supervision_to_samples(
    supervision, sampling_rate: int, max_samples: Optional[int] = None
) -> Tuple[int, int]:
    """
    Utility to convert a supervision's time span into a tuple of ``(start_sample num_samples)``.
    When ``max_samples`` is specified, it will truncate the ``num_samples`` (if necessary).
    """
    start_sample = compute_num_samples(supervision.start, sampling_rate=sampling_rate)
    num_samples = compute_num_samples(supervision.duration, sampling_rate=sampling_rate)
    if max_samples:
        diff = start_sample + num_samples - max_samples
        if diff > 0:
            num_samples -= diff
    return start_sample, num_samples


def is_none_or_gt(value, threshold) -> bool:
    return value is None or value > threshold


def is_equal_or_contains(
    value: Union[T, Sequence[T]], other: Union[T, Sequence[T]]
) -> bool:
    value = to_list(value)
    other = to_list(other)
    return set(other).issubset(set(value))


def is_module_available(*modules: str) -> bool:
    r"""Returns if a top-level module with :attr:`name` exists *without**
    importing it. This is generally safer than try-catch block around a
    `import X`. It avoids third party libraries breaking assumptions of some of
    our tests, e.g., setting multiprocessing start method when imported
    (see librosa/#747, torchvision/#544).

    Note: "borrowed" from torchaudio:
    https://github.com/pytorch/audio/blob/6bad3a66a7a1c7cc05755e9ee5931b7391d2b94c/torchaudio/_internal/module_utils.py#L9
    """
    import importlib

    return all(importlib.util.find_spec(m) is not None for m in modules)


def measure_overlap(lhs: Any, rhs: Any) -> float:
    """
    Given two objects with "start" and "end" attributes, return the % of their overlapped time
    with regard to the shorter of the two spans.
    ."""
    lhs, rhs = sorted([lhs, rhs], key=lambda item: item.start)
    overlapped_area = lhs.end - rhs.start
    if overlapped_area <= 0:
        return 0.0
    dur = min(lhs.end - lhs.start, rhs.end - rhs.start)
    return overlapped_area / dur


def ifnone(item: Optional[Any], alt_item: Any) -> Any:
    """Return ``alt_item`` if ``item is None``, otherwise ``item``."""
    return alt_item if item is None else item


def to_list(item: Union[Any, Sequence[Any]]) -> List[Any]:
    """Convert ``item`` to a list if it is not already a list."""
    return item if isinstance(item, list) else [item]


def to_hashable(item: Any) -> Any:
    """Convert ``item`` to a hashable type if it is not already hashable."""
    return tuple(item) if isinstance(item, list) else item


def hash_str_to_int(s: str, max_value: Optional[int] = None) -> int:
    """Hash a string to an integer in the range [0, max_value)."""
    if max_value is None:
        max_value = sys.maxsize
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16) % max_value


def lens_to_mask(lens: torch.IntTensor) -> torch.Tensor:
    """
    Create a 2-D mask tensor of shape (batch_size, max_length) and dtype float32
    from a 1-D tensor of integers describing the length of batch samples in another tensor.
    """
    mask = lens.new_zeros(lens.shape[0], max(lens), dtype=torch.float32)
    for i, num in enumerate(lens):
        mask[i, :num] = 1.0
    return mask


def rich_exception_info(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            raise type(e)(
                f"{e}\n[extra info] When calling: {fn.__qualname__}(args={args} kwargs={kwargs})"
            )

    return wrapper


class NonPositiveEnergyError(ValueError):
    pass


# Helper functions to mark a function as deprecated. The following is taken from:
# https://gist.github.com/kgriffs/8202106
class DeprecatedWarning(UserWarning):
    pass


def deprecated(message):
    """Flags a method as deprecated.
    Args:
        message: A human-friendly string of instructions, such
            as: 'Please migrate to add_proxy() ASAP.'
    """

    def decorator(func):
        """This is a decorator which can be used to mark functions
        as deprecated. It will result in a warning being emitted
        when the function is used."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            frame = inspect.currentframe().f_back

            warnings.warn_explicit(
                message,
                category=DeprecatedWarning,
                filename=inspect.getfile(frame.f_code),
                lineno=frame.f_lineno,
            )

            return func(*args, **kwargs)

        return wrapper

    return decorator


class suppress_and_warn:
    """Context manager to suppress specified exceptions that logs the error message.

    After the exception is suppressed, execution proceeds with the next
    statement following the with statement.

         >>> with suppress_and_warn(FileNotFoundError):
         ...     os.remove(somefile)
         >>> # Execution still resumes here if the file was already removed
    """

    def __init__(self, *exceptions, enabled: bool = True):
        self._enabled = enabled
        self._exceptions = exceptions

    def __enter__(self):
        pass

    def __exit__(self, exctype, excinst, exctb):
        if not self._enabled:
            return
        # Returning True from __exit__ in a context manager tells Python
        # to suppress an exception.
        should_suppress = exctype is not None and issubclass(exctype, self._exceptions)
        if should_suppress:
            logging.warning(
                f"[Suppressed {exctype.__qualname__}] Error message: {excinst}"
            )
        return should_suppress


def streaming_shuffle(
    data: Iterator[T],
    bufsize: int = 10000,
    rng: Optional[random.Random] = None,
) -> Generator[T, None, None]:
    """
    Shuffle the data in the stream.

    This uses a buffer of size ``bufsize``. Shuffling at
    startup is less random; this is traded off against
    yielding samples quickly.

    This code is mostly borrowed from WebDataset; note that we use much larger default
    buffer size because Cuts are very lightweight and fast to read.
    https://github.com/webdataset/webdataset/blob/master/webdataset/iterators.py#L145

    .. warning: The order of the elements is expected to be much less random than
        if the whole sequence was shuffled before-hand with standard methods like
        ``random.shuffle``.

    :param data: iterator
    :param bufsize: buffer size for shuffling
    :param rng: either random module or random.Random instance
    :return: a generator of cuts, shuffled on-the-fly.
    """
    if rng is None:
        rng = random
    buf = []
    startup = True
    for sample in data:
        if len(buf) < bufsize:
            try:
                buf.append(next(data))
            except StopIteration:
                pass
        if len(buf) > 0:
            k = rng.randint(0, len(buf) - 1)
            sample, buf[k] = buf[k], sample
        if startup and len(buf) < bufsize:
            buf.append(sample)
            continue
        startup = False
        yield sample
    for sample in buf:
        yield sample


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    from itertools import tee

    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


class Pipe:
    """Wrapper class for subprocess.Pipe.

    This class looks like a stream from the outside, but it checks
    subprocess status and handles timeouts with exceptions.
    This way, clients of the class do not need to know that they are
    dealing with subprocesses.

    Note: This class is based on WebDataset and modified here.
    Original source is in https://github.com/webdataset/webdataset

    :param *args: passed to `subprocess.Pipe`
    :param **kw: passed to `subprocess.Pipe`
    :param timeout: timeout for closing/waiting
    :param ignore_errors: don't raise exceptions on subprocess errors
    :param ignore_status: list of status codes to ignore
    """

    def __init__(
        self,
        *args,
        mode: str,
        timeout: float = 7200.0,
        ignore_errors: bool = False,
        ignore_status: Optional[List] = None,
        **kw,
    ):
        """Create an IO Pipe."""
        from subprocess import PIPE, Popen

        self.ignore_errors = ignore_errors
        # 0 => correct program exit
        # 141 => broken pipe (e.g. because the main program was terminated)
        self.ignore_status = [0, 141] + ifnone(ignore_status, [])
        self.timeout = timeout
        self.args = (args, kw)
        if mode[0] == "r":
            self.proc = Popen(*args, stdout=PIPE, text="b" not in mode, **kw)
            self.stream = self.proc.stdout
            if self.stream is None:
                raise ValueError(f"{args}: couldn't open")
        elif mode[0] == "w":
            self.proc = Popen(*args, stdin=PIPE, text="b" not in mode, **kw)
            self.stream = self.proc.stdin
            if self.stream is None:
                raise ValueError(f"{args}: couldn't open")
        self.status = None

    def __str__(self):
        return f"<Pipe {self.args}>"

    def check_status(self):
        """Poll the process and handle any errors."""
        status = self.proc.poll()
        if status is not None:
            self.wait_for_child()

    def is_running(self) -> bool:
        return self.proc.poll() is None

    def wait_for_child(self):
        """Check the status variable and raise an exception if necessary."""
        if self.status is not None:
            return
        self.status = self.proc.wait()
        if self.status not in self.ignore_status and not self.ignore_errors:
            raise Exception(f"{self.args}: exit {self.status} (read)")

    def read(self, *args, **kw):
        """Wrap stream.read and checks status."""
        result = self.stream.read(*args, **kw)
        self.check_status()
        return result

    def write(self, *args, **kw):
        """Wrap stream.write and checks status."""
        result = self.stream.write(*args, **kw)
        self.check_status()
        return result

    def readline(self, *args, **kw):
        """Wrap stream.readLine and checks status."""
        result = self.stream.readline(*args, **kw)
        self.status = self.proc.poll()
        self.check_status()
        return result

    def close(self):
        """Wrap stream.close, wait for the subprocess, and handle errors."""
        self.stream.close()
        self.status = self.proc.wait(self.timeout)
        self.wait_for_child()

    def __iter__(self):
        retval = self.readline()
        while retval:
            yield retval
            retval = self.readline()

    def __enter__(self):
        """Context handler."""
        return self

    def __exit__(self, etype, value, traceback):
        """Context handler."""
        self.close()


# Class to accept list of arguments as Click option
class PythonLiteralOption(click.Option):
    def type_cast_value(self, ctx, value):
        try:
            val = ast.literal_eval(value)
            if isinstance(val, list) or isinstance(val, tuple):
                return val[0] if len(val) == 1 else val
            else:
                return val
        except:
            return None


def is_torchaudio_available() -> bool:
    return is_module_available("torchaudio")


def build_rng(seed: Union[int, Literal["trng"]]) -> random.Random:
    if seed == "trng":
        return secrets.SystemRandom()
    else:
        return random.Random(seed)


_LHOTSE_DILL_ENABLED = False


def is_dill_enabled() -> bool:
    return _LHOTSE_DILL_ENABLED or os.environ["LHOTSE_DILL_ENABLED"]


def replace_bucket_with_profile_name(identifier, profile_name):
    parsed_identifier = urlparse(identifier)
    updated_identifier = parsed_identifier._replace(netloc=profile_name)
    return urlunparse(updated_identifier)
