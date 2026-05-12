"""
Binary index files for O(1) random-access reads into JSONL and tar archives.

Index File Format
-----------------
An index file stores an array of little-endian ``uint64`` byte-offsets,
one per sample, plus a sentinel entry equal to the file size.

For *N* samples the file is exactly ``(N + 1) * 8`` bytes:

    offset[0]  offset[1]  ...  offset[N-1]  file_size

Naming convention: ``<original_file>.idx``
    (e.g. ``cuts.000000.jsonl.idx``, ``recording.000000.tar.idx``)

Constraints
-----------
Index files **only work with uncompressed** data files:

* JSONL — plain ``.jsonl``, **not** ``.jsonl.gz``
* Tar   — plain ``.tar``,   **not** ``.tar.gz``

Usage
-----
>>> from lhotse.indexing import create_jsonl_index, IndexedJsonlReader
>>> create_jsonl_index("cuts.000000.jsonl")   # writes cuts.000000.jsonl.idx
>>> reader = IndexedJsonlReader("cuts.000000.jsonl")
>>> reader[42]   # O(1) dict from the 42nd line
"""

import hashlib
import io
import os
import struct
import tarfile
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional, Tuple, Union

if TYPE_CHECKING:
    from lhotse.array import Array, TemporalArray
    from lhotse.audio import Recording
    from lhotse.features import Features

import numpy as np

from lhotse.serialization import (
    decode_json_line,
    deserialize_item,
    extension_contains,
    open_best,
)
from lhotse.utils import Pathlike, is_valid_url

__all__ = [
    "create_jsonl_index",
    "create_tar_index",
    "create_shar_index",
    "index_file_path",
    "read_index",
    "index_exists",
    "supports_indexed_access",
    "validate_indexed_access",
    "LazyShuffledRange",
    "IndexedJsonlReader",
    "IndexedTarReader",
]

# ---------------------------------------------------------------------------
# Index path convention
# ---------------------------------------------------------------------------


def _path_str(path: Pathlike) -> str:
    return str(path)


def _is_pipe_path(path: Pathlike) -> bool:
    return _path_str(path).startswith("pipe:")


def _as_local_path(path: Pathlike) -> Optional[Path]:
    path_str = _path_str(path)
    if _is_pipe_path(path_str) or is_valid_url(path_str):
        return None
    return path if isinstance(path, Path) else Path(path_str)


def _is_compressed_path(path: Pathlike) -> bool:
    suffixes = Path(_path_str(path)).suffixes
    return bool(suffixes) and (
        suffixes[-1] in _COMPRESSED_SUFFIXES
        or (len(suffixes) >= 2 and suffixes[-2] in _COMPRESSED_SUFFIXES)
    )


def indexed_path_kind(path: Pathlike) -> Optional[str]:
    if _is_pipe_path(path) or _is_compressed_path(path):
        return None
    # Accept both ``.jsonl`` and ``.json`` for line-delimited manifests. NeMo
    # ships many ASR/SLM manifests as ``*.json`` (one JSON object per line,
    # despite the singular extension); ``create_jsonl_index`` and
    # ``IndexedJsonlReader`` only rely on newline-separated records, so the
    # storage layout is identical. A pretty-printed multi-line JSON would
    # produce a bogus index, but that's not a supported lhotse / NeMo
    # manifest layout in the first place.
    if extension_contains(".jsonl", path) or extension_contains(".json", path):
        return "jsonl"
    if extension_contains(".tar", path):
        return "tar"
    return None


def supports_indexed_access(path: Pathlike, *, kind: Optional[str] = None) -> bool:
    path_kind = indexed_path_kind(path)
    return path_kind is not None and (kind is None or path_kind == kind)


def validate_indexed_access(
    path: Pathlike,
    *,
    kind: Optional[str] = None,
    context: str = "Indexed reader",
) -> str:
    path_kind = indexed_path_kind(path)
    if path_kind is None:
        if _is_pipe_path(path):
            raise ValueError(
                f"{context} requires seekable data sources, but got a pipe command: {path}"
            )
        if _is_compressed_path(path):
            raise ValueError(
                f"{context} requires uncompressed JSONL or tar data, but got a compressed path: {path}"
            )
        raise ValueError(
            f"{context} requires a .jsonl or .tar data source, but got: {path}"
        )
    if kind is not None and path_kind != kind:
        raise ValueError(
            f"{context} expected a {kind} source, but got a {path_kind} path: {path}"
        )
    return path_kind


def index_file_path(data_path: Pathlike) -> Pathlike:
    """
    Return the conventional index file path for a given data file.

    Example::

        >>> index_file_path("cuts.000000.jsonl")
        PosixPath('cuts.000000.jsonl.idx')
    """
    local_path = _as_local_path(data_path)
    if local_path is not None:
        return Path(str(local_path) + ".idx")
    return str(data_path) + ".idx"


def index_exists(data_path: Pathlike, index_path: Optional[Pathlike] = None) -> bool:
    """
    Return ``True`` when a ``.idx`` file exists *and is usable*.

    When *index_path* is given, check that path instead of the conventional
    location next to *data_path*.

    A 0-byte or non-uint64-aligned file is treated as "not present" so
    callers re-trigger ``create_*_index`` instead of round-tripping an
    empty ``np.fromfile`` array through ``len(self._offsets) - 1`` and
    producing a negative ``__len__``. This guards against:
      * a previous writer crashing after the ``open("wb")`` truncate but
        before the payload write completed; and
      * a concurrent ``auto_create_index`` racer observing another worker's
        in-progress truncate (now also fixed at the write side via
        :func:`_write_index`'s stage-and-rename, but the size check stays
        as a belt-and-braces guard against stale 0-byte files on disk).
    """
    idx_path = index_path if index_path is not None else index_file_path(data_path)
    local_path = _as_local_path(idx_path)
    if local_path is not None:
        return _is_valid_index_file(local_path)
    try:
        with open_best(idx_path, "rb") as f:
            f.read(1)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Index I/O
# ---------------------------------------------------------------------------

_OFFSET_DTYPE = np.dtype("<u8")  # little-endian uint64


def _write_index(offsets: list, path: Pathlike) -> None:
    """Atomically write ``offsets`` to *path*.

    Stages bytes through a per-call tmp sibling and ``os.replace``s into
    place so concurrent readers / racing writers never observe a
    half-truncated 0-byte ``.idx``. The old non-atomic flow opened *path*
    in ``"wb"`` (which truncates immediately) and only filled it after
    walking the source — any reader that called ``index_exists +
    read_index`` between those two steps got back an empty ``np.fromfile``
    array and computed ``len(self._offsets) - 1 == -1``, crashing the
    sampler deep inside ``__len__``.
    """
    arr = np.array(offsets, dtype=_OFFSET_DTYPE)
    local_path = _as_local_path(path)
    if local_path is not None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        # ``os.replace`` is atomic on POSIX when src and dst live on the
        # same filesystem; placing the tmp file as a sibling guarantees
        # that. The pid + monotonic-ns suffix avoids cross-process
        # collisions when multiple ranks race to rebuild the same index.
        tmp_path = local_path.with_name(
            f"{local_path.name}.tmp.{os.getpid()}.{time.monotonic_ns()}"
        )
        try:
            with open(tmp_path, "wb") as f:
                f.write(arr.tobytes())
            os.replace(tmp_path, local_path)
        finally:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
        return
    # Non-local destination (e.g. ais:// / s3:// mirror tree) — there's no
    # atomic-rename primitive across most object stores, and concurrent
    # racing writers aren't a concern in those workflows (the .idx mirror
    # is built once by submit_build_indexes.py). Fall through to the
    # original direct-write path.
    with open_best(path, "wb") as f:
        f.write(arr.tobytes())


def _remote_index_cache_dir() -> Path:
    return Path(tempfile.gettempdir()) / "lhotse-index-cache"


def _remote_index_cache_path(idx_path: Pathlike) -> Path:
    digest = hashlib.sha256(_path_str(idx_path).encode("utf-8")).hexdigest()
    return _remote_index_cache_dir() / f"{digest}.idx"


def _is_valid_index_file(path: Path) -> bool:
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return False
    return size >= _OFFSET_DTYPE.itemsize and size % _OFFSET_DTYPE.itemsize == 0


def _materialize_remote_index(idx_path: Pathlike) -> Path:
    cache_path = _remote_index_cache_path(idx_path)
    if _is_valid_index_file(cache_path):
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f"{cache_path.name}.",
        suffix=".tmp",
        dir=str(cache_path.parent),
    )
    tmp_path = Path(tmp_name)

    try:
        with open_best(idx_path, "rb") as src, os.fdopen(fd, "wb") as dst:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
            dst.flush()
            os.fsync(dst.fileno())
        if not _is_valid_index_file(tmp_path):
            raise FileNotFoundError(
                f"Index file not found, empty, or invalid: {idx_path}"
            )
        os.replace(tmp_path, cache_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return cache_path


def read_index(idx_path: Pathlike) -> np.ndarray:
    """
    Read an index file and return a NumPy array of byte-offsets.

    Indexes are tiny (a uint64 per record + a sentinel; typical shards land
    in the 1 KB – 100 KB range), so ``np.fromfile`` is preferred over
    ``np.memmap``: at large fan-outs (NeMo blends with 80k+ shards), the per
    -mmap address-space slot consumes the kernel's ``vm.max_map_count``
    budget (~65k by default) and any subsequent ``mmap`` raises
    ``OSError: [Errno 12] Cannot allocate memory``. Resident memory cost of
    the alternative is negligible (a few hundred MB for the entire training
    blend), and lookups are just integer indexing — no benefit from memmap
    semantics. Remote index files are cached under a deterministic local
    temp path and read from there. The last element is the sentinel
    (file size); there are ``len(arr) - 1`` samples.
    """
    local_path = _as_local_path(idx_path)
    if local_path is not None:
        if not local_path.is_file():
            raise FileNotFoundError(f"Index file not found: {local_path}")
        return np.fromfile(local_path, dtype=_OFFSET_DTYPE)
    cache_path = _materialize_remote_index(idx_path)
    return np.fromfile(cache_path, dtype=_OFFSET_DTYPE)


# ---------------------------------------------------------------------------
# Index creation
# ---------------------------------------------------------------------------


def create_jsonl_index(
    jsonl_path: Pathlike, output_path: Optional[Pathlike] = None
) -> Path:
    """
    Scan an **uncompressed** JSONL file and build a binary index.

    Each entry in the index is the byte-offset of the corresponding line's
    first character.  A final sentinel entry stores the file size.

    :param output_path: if set, write the ``.idx`` file to this path
        instead of the conventional location next to *jsonl_path*.
    :returns: the path of the newly created ``.idx`` file.
    """
    _assert_uncompressed(jsonl_path, "JSONL")
    # Track the running byte offset by accumulating ``len(line)`` rather than
    # calling ``f.tell()`` on every line. The latter raises
    # ``io.UnsupportedOperation`` on non-seekable streams (AIStore's
    # ``ObjectFileReader``, smart_open's S3 reader without seek support, …),
    # which would make every ``s3://``/``ais://`` JSONL fail to index.
    # ``readline()`` returns the bytes including the trailing newline, so the
    # accumulated total exactly tracks the start-of-line byte offsets.
    offsets = []
    pos = 0
    with open_best(jsonl_path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                break
            offsets.append(pos)
            pos += len(line)
        offsets.append(pos)  # sentinel = file size

    idx_path = output_path if output_path is not None else index_file_path(jsonl_path)
    _write_index(offsets, idx_path)
    return idx_path


def create_tar_index(
    tar_path: Pathlike, output_path: Optional[Pathlike] = None
) -> Path:
    """
    Scan an **uncompressed** tar archive and build a binary index.

    Each entry records the byte-offset of the **first** tar header in each
    sample pair (data member + metadata member), following the Lhotse Shar
    convention where every sample consists of exactly two consecutive tar
    members.

    A final sentinel entry stores the file size.

    :param output_path: if set, write the ``.idx`` file to this path
        instead of the conventional location next to *tar_path*.
    :returns: the path of the newly created ``.idx`` file.
    """
    _assert_uncompressed(tar_path, "tar")

    offsets = []
    num_members = 0
    # Sentinel needs to be ≥ "byte position past the last sample pair's
    # data" so ``IndexedTarReader.member_byte_range`` can recover the upper
    # bound for the final sample. Prefer ``f.tell()`` on the underlying
    # handle when supported (= bytes consumed from storage, typically the
    # file size since the tar record buffer reads in 10 KiB chunks) so the
    # written .idx matches the historical "sentinel = file size" semantics.
    # Non-seekable streams (e.g. AIS ``ObjectFileReader``) inherit
    # ``BufferedIOBase.tell()``, which delegates to ``seek(0, SEEK_CUR)`` and
    # raises ``UnsupportedOperation: seek``; fall back to ``tf.offset``,
    # which ``tarfile`` itself tracks and which sits at the start of the
    # trailing EOF-marker block once iteration stops on ``EOFHeaderError``.
    with open_best(tar_path, "rb") as f:
        with tarfile.open(fileobj=f, mode="r|") as tf:
            for member in tf:
                if num_members % 2 == 0:
                    offsets.append(member.offset)
                num_members += 1
            sentinel_from_tarfile = tf.offset
        if num_members % 2 != 0:
            raise RuntimeError(
                f"Expected an even number of tar members (data+meta pairs) "
                f"in {tar_path}, got {num_members}."
            )
        try:
            sentinel = f.tell()
        except (io.UnsupportedOperation, OSError, AttributeError):
            sentinel = sentinel_from_tarfile
        offsets.append(sentinel)

    idx_path = output_path if output_path is not None else index_file_path(tar_path)
    _write_index(offsets, idx_path)
    return idx_path


def create_shar_index(
    shar_dir: Pathlike, output_dir: Optional[Pathlike] = None
) -> None:
    """
    Create binary index files for **all** JSONL and tar files in a
    Shar directory.

    Compressed files (``.gz``) are silently skipped because they cannot
    be indexed.

    :param output_dir: if set, write ``.idx`` files into this directory
        (using the same filenames as the conventional location, but under
        a different parent).
    """
    shar_dir = Path(shar_dir)
    for p in sorted(shar_dir.iterdir()):
        out = None
        if output_dir is not None:
            out = Path(output_dir) / (p.name + ".idx")
        if p.suffix == ".jsonl":
            create_jsonl_index(p, output_path=out)
        elif p.suffix == ".tar":
            create_tar_index(p, output_path=out)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COMPRESSED_SUFFIXES = {".gz", ".bz2", ".xz", ".lz4", ".zst"}


def _assert_uncompressed(path: Pathlike, kind: str) -> None:
    if _is_compressed_path(path):
        raise RuntimeError(
            f"Cannot create an index for a compressed {kind} file: {path}. "
            f"Only uncompressed files are supported."
        )


# ---------------------------------------------------------------------------
# LazyShuffledRange — O(1) memory permutation via Feistel cipher
# ---------------------------------------------------------------------------


class LazyShuffledRange:
    """
    An O(1)-memory lazy permutation of ``range(n)`` determined by *seed*.

    Uses a balanced Feistel network (6 rounds) with cycle-walking to
    handle non-power-of-two domain sizes.  Every element of ``[0, n)``
    is produced exactly once in a deterministic, seed-dependent order.

    When ``num_shards > 1``, this object yields only the subset of the
    permutation belonging to shard ``shard_id``: it walks logical offsets
    ``shard_id, shard_id + num_shards, shard_id + 2*num_shards, ...`` and
    applies the Feistel permutation to each. Across all ``num_shards``
    shards, every element of ``[0, n)`` is produced exactly once — making
    this the single primitive for DP-rank / worker-id data partitioning
    in the iterable-dataset path. Defaults ``(shard_id=0, num_shards=1)``
    reproduce the un-sharded behaviour bitwise.

    Supports checkpointing via :meth:`state_dict` / :meth:`load_state_dict`.

    Example::

        >>> perm = LazyShuffledRange(1000, seed=42)
        >>> perm[0]   # first element of the permutation — O(1)
        >>> list(perm) == sorted(range(1000))  # after sorting, same elements
        True
    """

    def __init__(
        self,
        n: int,
        seed: int,
        shard_id: int = 0,
        num_shards: int = 1,
    ) -> None:
        if num_shards < 1:
            raise ValueError(f"num_shards must be >= 1, got {num_shards}")
        if not (0 <= shard_id < num_shards):
            raise ValueError(
                f"shard_id must be in [0, num_shards={num_shards}), got {shard_id}"
            )
        self.n = n
        self.seed = seed
        self.shard_id = shard_id
        self.num_shards = num_shards
        self._pos = 0

        # Compute the number of bits for each half of the Feistel network.
        # We need total_bits such that 2**total_bits >= n.
        if n <= 1:
            self._half_bits = 1
            self._total_bits = 2
        else:
            self._total_bits = max(2, (n - 1).bit_length())
            # Make total_bits even
            if self._total_bits % 2 != 0:
                self._total_bits += 1
            self._half_bits = self._total_bits // 2

        self._half_mask = (1 << self._half_bits) - 1

        # Pre-compute round keys from the seed.
        self._num_rounds = 6
        rng = np.random.RandomState(seed & 0xFFFFFFFF)
        self._round_keys = [
            int(rng.randint(0, 2**63)) for _ in range(self._num_rounds)
        ]

    def __len__(self) -> int:
        if self.n <= self.shard_id:
            return 0
        return (self.n - self.shard_id + self.num_shards - 1) // self.num_shards

    def __getitem__(self, idx: int) -> int:
        """Return the *idx*-th element of this shard's permutation in O(1)."""
        shard_len = len(self)
        if idx < 0:
            idx += shard_len
        if idx < 0 or idx >= shard_len:
            raise IndexError(
                f"index {idx} out of range for LazyShuffledRange(n={self.n}, "
                f"shard_id={self.shard_id}, num_shards={self.num_shards}) "
                f"with shard length {shard_len}"
            )
        logical_offset = self.shard_id + idx * self.num_shards
        return self._permute(logical_offset)

    def __iter__(self) -> "LazyShuffledRange":
        return self

    def __next__(self) -> int:
        logical_offset = self.shard_id + self._pos * self.num_shards
        if logical_offset >= self.n:
            raise StopIteration
        val = self._permute(logical_offset)
        self._pos += 1
        return val

    def reset(self) -> None:
        """Reset the iterator to the beginning of this shard."""
        self._pos = 0

    def state_dict(self) -> dict:
        """Return checkpoint state: ``{"n", "seed", "shard_id", "num_shards", "pos"}``."""
        return {
            "n": self.n,
            "seed": self.seed,
            "shard_id": self.shard_id,
            "num_shards": self.num_shards,
            "pos": self._pos,
        }

    def load_state_dict(self, sd: dict) -> None:
        """Restore from a checkpoint produced by :meth:`state_dict`.

        Validates the full topology — ``n``, ``seed``, ``shard_id``, and
        ``num_shards`` must match the current instance. A mismatch is a
        loud error (elastic resume with a different DP/worker topology
        is out of scope; the user must drop dataloader state).
        Legacy state_dicts without shard fields are interpreted as
        ``(shard_id=0, num_shards=1)`` and load only into matching contexts.
        """
        saved_shard_id = sd.get("shard_id", 0)
        saved_num_shards = sd.get("num_shards", 1)
        if (
            sd["n"] != self.n
            or sd["seed"] != self.seed
            or saved_shard_id != self.shard_id
            or saved_num_shards != self.num_shards
        ):
            raise ValueError(
                f"LazyShuffledRange state mismatch: "
                f"expected n={self.n}, seed={self.seed}, "
                f"shard_id={self.shard_id}, num_shards={self.num_shards}; "
                f"got n={sd['n']}, seed={sd['seed']}, "
                f"shard_id={saved_shard_id}, num_shards={saved_num_shards}. "
                f"Resuming with a different DP/worker topology is not supported — "
                f"drop dataloader state if the topology changed."
            )
        self._pos = sd["pos"]

    # ---- Feistel internals ------------------------------------------------

    def _round_fn(self, value: int, round_key: int) -> int:
        """Knuth multiplicative hash used as the Feistel round function."""
        # 64-bit Knuth multiplicative hash
        h = ((value ^ round_key) * 2654435761) & 0xFFFFFFFFFFFFFFFF
        # Mix bits
        h ^= h >> 17
        h = (h * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        h ^= h >> 31
        return h & self._half_mask

    def _feistel(self, x: int) -> int:
        """One pass of the balanced Feistel cipher."""
        left = (x >> self._half_bits) & self._half_mask
        right = x & self._half_mask
        for i in range(self._num_rounds):
            new_left = right
            new_right = left ^ self._round_fn(right, self._round_keys[i])
            left, right = new_left, new_right
        return (left << self._half_bits) | right

    def _permute(self, idx: int) -> int:
        """Map *idx* to a unique element in ``[0, n)`` using cycle-walking."""
        x = idx
        while True:
            x = self._feistel(x)
            if x < self.n:
                return x


# ---------------------------------------------------------------------------
# IndexedJsonlReader — O(1) random-access JSONL reader
# ---------------------------------------------------------------------------


class IndexedJsonlReader:
    """
    Random-access reader for an uncompressed JSONL file using a binary
    index.

    Each ``__getitem__`` call performs a single seek + readline + JSON parse,
    giving O(1) access to any line.

    Parameters
    ----------
    path : Pathlike
        Path to the uncompressed JSONL file.
    auto_create_index : bool
        If ``True`` (default), the ``.idx`` file will be created
        automatically when it is missing.  Set to ``False`` to raise
        :class:`FileNotFoundError` instead.
    index_path : Pathlike, optional
        Custom path to the ``.idx`` file.  When set, this path is used
        instead of the conventional ``<path>.idx`` location.  Useful when
        the data file lives on a remote object store but the index must
        be on local disk.
    """

    def __init__(
        self,
        path: Pathlike,
        auto_create_index: bool = True,
        index_path: Optional[Pathlike] = None,
    ) -> None:
        validate_indexed_access(path, kind="jsonl", context="IndexedJsonlReader")
        self.path = path
        self.index_path = index_path
        self._fh: Optional[object] = None
        idx_path = (
            self.index_path
            if self.index_path is not None
            else index_file_path(self.path)
        )
        if not index_exists(self.path, index_path=idx_path):
            if auto_create_index:
                create_jsonl_index(self.path, output_path=idx_path)
            else:
                raise FileNotFoundError(
                    f"Index file not found: {idx_path}. "
                    f"Use create_jsonl_index() to build it, or set auto_create_index=True."
                )
        self._offsets = read_index(idx_path)

    def _ensure_open(self):
        if self._fh is None:
            self._fh = open_best(self.path, "rb")

    def __del__(self):
        self.close()

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_fh"] = None  # file handles are not picklable
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __len__(self) -> int:
        return len(self._offsets) - 1  # last entry is the sentinel

    def __getitem__(self, idx: int) -> dict:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"index {idx} out of range for IndexedJsonlReader with {len(self)} lines"
            )
        self._ensure_open()
        start = int(self._offsets[idx])
        end = int(self._offsets[idx + 1])
        self._fh.seek(start)
        line = self._fh.read(end - start)
        return decode_json_line(line.decode("utf-8"))

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


# ---------------------------------------------------------------------------
# IndexedTarReader — O(1) random-access tar reader for Shar format
# ---------------------------------------------------------------------------

# Tar block size
_TAR_BLOCK_SIZE = 512


def _ceil_block(size: int) -> int:
    """Round *size* up to the next 512-byte boundary."""
    return (size + _TAR_BLOCK_SIZE - 1) // _TAR_BLOCK_SIZE * _TAR_BLOCK_SIZE


Manifest = Union["Recording", "Array", "TemporalArray", "Features"]


class IndexedTarReader:
    """
    Random-access reader for an uncompressed Lhotse Shar tar archive
    using a binary index.

    Each sample in the tar file consists of two consecutive members
    (data + metadata).  ``__getitem__`` seeks to the pair's first header,
    reads both members, and returns the same ``(manifest, data_path)``
    tuple as :class:`~lhotse.shar.readers.tar.TarIterator`.

    Parameters
    ----------
    path : Pathlike
        Path to the uncompressed tar archive.
    auto_create_index : bool
        If ``True`` (default), the ``.idx`` file will be created when
        missing.
    index_path : Pathlike, optional
        Custom path to the ``.idx`` file.  When set, this path is used
        instead of the conventional ``<path>.idx`` location.
    """

    def __init__(
        self,
        path: Pathlike,
        auto_create_index: bool = True,
        index_path: Optional[Pathlike] = None,
    ) -> None:
        validate_indexed_access(path, kind="tar", context="IndexedTarReader")
        self.path = path
        self.index_path = index_path
        self._fh: Optional[object] = None
        idx_path = (
            self.index_path
            if self.index_path is not None
            else index_file_path(self.path)
        )
        if not index_exists(self.path, index_path=idx_path):
            if auto_create_index:
                create_tar_index(self.path, output_path=idx_path)
            else:
                raise FileNotFoundError(
                    f"Index file not found: {idx_path}. "
                    f"Use create_tar_index() to build it, or set auto_create_index=True."
                )
        self._offsets = read_index(idx_path)

    def _ensure_open(self):
        if self._fh is None:
            self._fh = open_best(self.path, "rb")

    def __del__(self):
        self.close()

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_fh"] = None  # file handles are not picklable
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __len__(self) -> int:
        return len(self._offsets) - 1

    def __getitem__(self, idx: int) -> Tuple[Optional[Manifest], Path]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"index {idx} out of range for IndexedTarReader with {len(self)} samples"
            )
        self._ensure_open()

        offset = int(self._offsets[idx])

        # Read first member (data)
        data, data_path = self._read_member(offset)
        # Advance past data to next header
        next_offset = offset + _TAR_BLOCK_SIZE + _ceil_block(self._last_member_size)
        # Read second member (metadata)
        meta_bytes, meta_path = self._read_member(next_offset)

        # Process like TarIterator
        if meta_bytes is not None:
            from lhotse.shar.utils import fill_shar_placeholder

            meta = deserialize_item(decode_json_line(meta_bytes.decode("utf-8")))
            fill_shar_placeholder(manifest=meta, data=data, tarpath=data_path)
        else:
            meta = None

        return meta, data_path

    def _read_member(self, offset: int) -> Tuple[Optional[bytes], Path]:
        """Read a single tar member at the given byte offset."""
        self._fh.seek(offset)
        data, path, info = read_tar_member_at(self._fh, offset)
        self._last_member_size = info.size
        return data, path

    def member_byte_range(self, idx: int) -> Tuple[int, int]:
        """Return ``(offset, end_offset)`` for the *idx*-th sample-pair, where
        ``end_offset`` is the first byte beyond this sample's contiguous block
        (= the next sample's offset, or the file size for the last sample).
        """
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"index {idx} out of range for IndexedTarReader with {len(self)} samples"
            )
        return int(self._offsets[idx]), int(self._offsets[idx + 1])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def read_tar_member_at(fh, offset: int) -> Tuple[Optional[bytes], Path, tarfile.TarInfo]:
    """Read a single tar member's header + payload at ``offset`` from an open
    file handle. Returns ``(data_bytes, member_path, tar_info)``.

    ``data_bytes`` is None for ``.nodata``/``.nometa`` placeholder members.
    Does NOT validate type or skip non-regular members — pass an offset that
    points at a regular file's header.
    """
    fh.seek(offset)
    header_buf = fh.read(_TAR_BLOCK_SIZE)
    if len(header_buf) < _TAR_BLOCK_SIZE:
        raise RuntimeError(f"Unexpected EOF reading tar header at offset {offset}")
    info = tarfile.TarInfo.frombuf(header_buf, tarfile.ENCODING, "surrogateescape")
    path = Path(info.name)
    if path.suffix in (".nodata", ".nometa"):
        return None, path, info
    data = fh.read(info.size)
    return data, path, info
