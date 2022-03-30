import functools
import inspect
import logging
import math
import random
import uuid
import warnings
from contextlib import AbstractContextManager, contextmanager
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_HALF_DOWN, ROUND_HALF_UP
from math import ceil, isclose
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import torch
from tqdm.auto import tqdm
from typing_extensions import Literal

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
    compression: Optional[str] = None
    import_err_msg = (
        "Please do 'pip install smart_open' - "
        "if you are using S3/GCP/Azure/other cloud-specific URIs, do "
        "'pip install smart_open[s3]' (or smart_open[gcp], etc.) instead."
    )
    smart_open: Optional[Callable] = None

    @classmethod
    def setup(
        cls, compression: Optional[str] = None, transport_params: Optional[dict] = None
    ):
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
        if cls.compression is not None and cls.compression != compression:
            logging.warning(
                f"SmartOpen.setup second call overwrites existing compression param with new version"
                f"\t\n{cls.compression} vs {compression}"
            )
        cls.transport_params = transport_params
        cls.compression = compression
        cls.smart_open = sm_open

    @classmethod
    def open(cls, uri, mode="rb", compression=None, transport_params=None, **kwargs):
        if cls.smart_open is None:
            cls.setup(compression=compression, transport_params=transport_params)
        compression = compression if compression else cls.compression
        transport_params = (
            transport_params if transport_params else cls.transport_params
        )
        return cls.smart_open(
            uri,
            mode=mode,
            compression=compression,
            transport_params=transport_params,
            **kwargs,
        )


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


# TODO: Ugh, Protocols are only in Python 3.8+...
def overlaps(lhs: Any, rhs: Any) -> bool:
    """Indicates whether two time-spans/segments are overlapping or not."""
    return (
        lhs.start < rhs.end
        and rhs.start < lhs.end
        and not isclose(lhs.start, rhs.end)
        and not isclose(rhs.start, lhs.end)
    )


def overspans(spanning: Any, spanned: Any) -> bool:
    """Indicates whether the left-hand-side time-span/segment covers the whole right-hand-side time-span/segment."""
    return spanning.start <= spanned.start <= spanned.end <= spanning.end


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
    it: Iterable[Any], output_dir: Pathlike, chunk_size: int
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
        Each manifest is saved at: ``{output_dir}/{split_idx}.jsonl.gz``
    :param chunk_size: the number of items in each chunk.
    :return: a list of lazily opened chunk manifests.
    """
    from lhotse.serialization import SequentialJsonlWriter

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    items = iter(it)
    split_idx = 0
    splits = []
    while True:
        try:
            written = 0
            with SequentialJsonlWriter(output_dir / f"{split_idx}.jsonl.gz") as writer:
                while written < chunk_size:
                    item = next(items)
                    writer.write(item)
                    written += 1
            split_idx += 1
        except StopIteration:
            break
        finally:
            splits.append(writer.open_manifest())

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

    with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=desc) as t:
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
    return int(
        Decimal(round(num_samples / factor, ndigits=8)).quantize(0, rounding=rounding)
    )


def compute_num_samples(
    duration: Seconds, sampling_rate: int, rounding=ROUND_HALF_UP
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


def add_durations(d1: Seconds, d2: Seconds, sampling_rate: int) -> Seconds:
    """
    Adds two durations in a way that avoids floating point precision issues.
    The durations in seconds are first converted to audio sample counts,
    then added, and finally converted back to floating point seconds.
    """
    s1 = compute_num_samples(d1, sampling_rate=sampling_rate)
    s2 = compute_num_samples(d2, sampling_rate=sampling_rate)
    tot = s1 + s2
    return tot / sampling_rate


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


def index_by_id_and_check(manifests: Iterable[T]) -> Dict[str, T]:
    id2man = {}
    for m in manifests:
        assert m.id not in id2man, f"Duplicated manifest ID: {m.id}"
        id2man[m.id] = m
    return id2man


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
    data: Iterable[T],
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
        k = rng.randint(0, len(buf) - 1)
        sample, buf[k] = buf[k], sample
        if startup and len(buf) < bufsize:
            buf.append(sample)
            continue
        startup = False
        yield sample
    for sample in buf:
        yield sample
