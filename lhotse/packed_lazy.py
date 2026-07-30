"""
Lazy manifest iteration backed by :mod:`lhotse.index_pack`.

The iterator in this module replaces a chain of per-shard indexed readers with
one logical collection stored in an ``.idxpack``. It retains random access,
worker partitioning, deterministic global shuffling, and checkpoint/resume
semantics while avoiding eager shard-path expansion, one reader object per
shard, and one in-memory offset array per sidecar.
"""

from __future__ import annotations

import os
import warnings
import weakref
from collections import OrderedDict
from collections.abc import Callable
from json import JSONDecodeError
from typing import Any

from lhotse.index_pack import IndexPack, open_index_pack
from lhotse.lazy import (
    IteratorNode,
    attach_graph_origin,
    normalize_graph_token,
    resolve_iteration_seed,
)
from lhotse.serialization import decode_json_line, deserialize_item
from lhotse.utils import is_valid_url


def read_packed_range(
    index_pack: IndexPack,
    path: str,
    start: int,
    end: int,
    *,
    max_open_files: int = 32,
) -> bytes:
    """
    Read an exact local byte range through a pack-shared descriptor cache.

    All readers using the same :class:`~lhotse.index_pack.IndexPack` share one
    process-local LRU, so ``max_open_files`` is a bound per dataset pack rather
    than per logical collection. Remote source URLs are rejected because
    ``os.pread()`` requires a local, seekable file.
    """
    cache = _file_cache_for_pack(index_pack, max_open_files)
    return cache.read(path, start, end)


class LazyPackedManifestIterator(IteratorNode):
    """
    Lazily decode an ordered sharded manifest collection from an ``.idxpack``.

    A conventional large sharded dataset is often represented as a
    ``LazyIteratorChain`` containing one ``LazyIndexedManifestIterator`` per
    file. Constructing that graph expands every path and loads or maps every
    sidecar separately. This iterator presents the same shards as one virtual
    sequence: the pack mmap resolves a logical index to ``(path, start, end)``,
    and only that record is read with ``pread()`` through a bounded descriptor
    cache.

    Integer lookup addresses the virtual concatenation. A
    ``(shard_index, local_index)`` tuple addresses one record in one shard and
    preserves graph-origin tokens expected by Lhotse transforms. Sequential
    iteration partitions records within each shard across data-loader workers;
    ``shuffle_shards=True`` uses Lhotse's deterministic lazy permutation across
    the entire collection. Both modes support :meth:`state_dict` and
    :meth:`load_state_dict`.

    Args:
        index_pack:
            An :class:`~lhotse.index_pack.IndexPack` or local pack path. Paths
            use the process-local mapping cache returned by
            :func:`~lhotse.index_pack.open_index_pack`.
        collection_key:
            The collection's 32-byte identity or hexadecimal representation.
        shuffle_shards:
            Apply a deterministic lazy global permutation on each iteration.
            The name matches the corresponding sharded-manifest configuration
            option.
        seed:
            Base seed used to derive each iteration's permutation.
        decode:
            Optional callable applied to the dictionary decoded from each JSON
            line. The default is :func:`lhotse.serialization.deserialize_item`.
        skip_decode_errors:
            Skip malformed UTF-8 or JSON records instead of raising.
        decode_error_callback:
            Optional ``callback(exception, global_index, path)`` invoked for
            each skipped record.
        max_open_files:
            Maximum number of source files held open by the descriptor cache
            shared by all iterators reading this pack in the current process.

    Example::

        from lhotse.index_pack import index_pack_collection_key

        key = index_pack_collection_key(
            role="records",
            kind="json-lines",
            source_spec="cuts-{000..127}.jsonl",
        )
        source = LazyPackedManifestIterator(
            "dataset.idxpack",
            key,
            shuffle_shards=True,
            seed=42,
        )
        first_cut = next(iter(source))
    """

    is_checkpointable = True
    is_indexed = True
    has_constant_time_access = True

    def __init__(
        self,
        index_pack,
        collection_key: bytes | str,
        *,
        shuffle_shards: bool = False,
        seed: int = 0,
        decode: Callable[[dict], Any] | None = None,
        skip_decode_errors: bool = False,
        decode_error_callback: Callable[[BaseException, int, str], None] | None = None,
        max_open_files: int = 32,
    ):
        self.index_pack = (
            index_pack
            if isinstance(index_pack, IndexPack)
            else open_index_pack(index_pack)
        )
        self.collection_key = collection_key
        self.collection = self.index_pack.collection(collection_key)
        self.shuffle_shards = shuffle_shards
        self.seed = seed
        self._decode = decode if decode is not None else deserialize_item
        self.skip_decode_errors = skip_decode_errors
        self.decode_error_callback = decode_error_callback
        if max_open_files < 1:
            raise ValueError("max_open_files must be positive")
        self.max_open_files = max_open_files

        self.num_iters = 0
        self._current_shard = 0
        self._current_position = 0
        self._global_position = 0
        self._global_seed = None
        self._shard_id = None
        self._num_shards = None
        self._restored = False

    def __len__(self) -> int:
        """Return the total number of records in the packed collection."""
        return len(self.collection)

    def __getitem__(self, token):
        """
        Decode one record by global index or ``(shard, local_index)`` token.

        Negative global, shard, and local indices follow Python sequence
        semantics.
        """
        return self._decode_token(token)

    def read_with_location(self, token):
        """
        Decode one record and return it together with its packed byte location.

        This avoids resolving the same token twice in adapters that also need
        the manifest shard/local position to locate an associated payload.
        """
        normalized_token, global_index, location = self._location_for_token(token)
        raw = read_packed_range(
            self.index_pack,
            location.path,
            location.start,
            location.end,
            max_open_files=self.max_open_files,
        )
        decoded_line = raw.decode("utf-8")
        try:
            item = self._decode(decode_json_line(decoded_line))
        except JSONDecodeError as ex:
            preview = decoded_line[:120].replace("\n", "\\n").replace("\r", "\\r")
            msg = (
                f"{ex.msg} while decoding packed JSONL record "
                f"path={location.path!r} pack={str(self.index_pack.path)!r} "
                f"idx={global_index} byte_range=[{location.start}, {location.end}) "
                f"preview={preview!r}"
            )
            raise JSONDecodeError(msg, ex.doc, ex.pos) from ex
        return attach_graph_origin(item, normalized_token), location

    def __iter__(self):
        """Iterate using the configured deterministic ordering and worker partition."""
        if self.shuffle_shards:
            return self._iter_globally_shuffled()
        return self._iter_sequential()

    def state_dict(self) -> dict:
        """
        Return the resumable iterator state.

        The common keys intentionally match ``LazyIteratorChain`` where
        possible; ``packed_current_position`` records the within-shard
        position needed by sequential iteration.
        """
        return {
            "current_iter_idx": self._current_shard,
            "num_iters": self.num_iters,
            "iter_order": None,
            "global_position": self._global_position,
            "global_seed": self._global_seed,
            "global_shard_id": self._shard_id,
            "global_num_shards": self._num_shards,
            "packed_current_position": self._current_position,
        }

    def load_state_dict(self, state: dict) -> None:
        """
        Restore a state produced by :meth:`state_dict`.

        Worker partition compatibility is checked when iteration resumes so a
        checkpoint cannot silently continue under a different partition.
        """
        self._current_shard = state.get("current_iter_idx", 0)
        self._current_position = state.get("packed_current_position", 0)
        self.num_iters = state.get("num_iters", 0)
        self._global_position = state.get("global_position", 0)
        self._global_seed = state.get("global_seed")
        self._shard_id = state.get("global_shard_id")
        self._num_shards = state.get("global_num_shards")
        self._restored = True

    def close(self) -> None:
        """
        Release no resources.

        Pack mappings and source descriptors are shared across iterators and
        are reclaimed when the owning :class:`IndexPack` is no longer used.
        """
        return

    def _location_for_token(self, token):
        normalized_token = normalize_graph_token(token)
        if isinstance(normalized_token, tuple) and len(normalized_token) == 2:
            shard_index, local_index = normalized_token
            location = self.collection.locate_in_shard(shard_index, local_index)
            previous_end = (
                self.index_pack._sequence(
                    self.collection.sequence_start + location.shard_index - 1
                )[1]
                if location.shard_index
                else 0
            )
            return normalized_token, previous_end + location.local_index, location
        if not isinstance(normalized_token, int):
            raise TypeError(
                f"Unsupported packed manifest graph token: {normalized_token!r}"
            )
        global_index = normalized_token
        if global_index < 0:
            global_index += len(self.collection)
        return normalized_token, global_index, self.collection.locate(global_index)

    def _decode_token(self, token):
        item, _ = self.read_with_location(token)
        return item

    def _decode_or_skip(self, token):
        try:
            return self._decode_token(token)
        except (JSONDecodeError, UnicodeDecodeError) as ex:
            if not self.skip_decode_errors:
                raise
            _, global_index, location = self._location_for_token(token)
            if self.decode_error_callback is not None:
                self.decode_error_callback(ex, global_index, location.path)
            else:
                warnings.warn(
                    f"Skipping malformed packed manifest record "
                    f"{global_index} in {location.path}: {ex}",
                    stacklevel=2,
                )
            return None

    def _iter_globally_shuffled(self):
        from lhotse.dataset.dataloading import get_worker_partition
        from lhotse.indexing import LazyShuffledRange

        shard_id, num_shards = get_worker_partition()
        if self._restored:
            self._restored = False
            start = self._global_position
            base_seed = self._global_seed
            if base_seed is None:
                base_seed = resolve_iteration_seed(self.seed)
            if self._num_shards is not None and (
                self._shard_id != shard_id or self._num_shards != num_shards
            ):
                raise ValueError(
                    "LazyPackedManifestIterator partition mismatch on resume: "
                    f"saved (shard_id={self._shard_id}, num_shards={self._num_shards}), "
                    f"current (shard_id={shard_id}, num_shards={num_shards})."
                )
        else:
            start = 0
            self._global_position = 0
            base_seed = resolve_iteration_seed(self.seed)
            self._global_seed = base_seed
        self._shard_id = shard_id
        self._num_shards = num_shards

        shuffled = LazyShuffledRange(
            len(self),
            seed=base_seed + self.num_iters,
            shard_id=shard_id,
            num_shards=num_shards,
        )
        for position in range(start, len(shuffled)):
            self._global_position = position + 1
            token = shuffled[position]
            item = self._decode_or_skip(token)
            if item is not None:
                yield item
        self.num_iters += 1

    def _iter_sequential(self):
        from lhotse.dataset.dataloading import get_worker_partition

        shard_id, num_shards = get_worker_partition()
        if self._restored:
            self._restored = False
            start_shard = self._current_shard
            start_position = self._current_position
            if self._num_shards is not None and (
                self._shard_id != shard_id or self._num_shards != num_shards
            ):
                raise ValueError(
                    "LazyPackedManifestIterator partition mismatch on resume: "
                    f"saved (shard_id={self._shard_id}, num_shards={self._num_shards}), "
                    f"current (shard_id={shard_id}, num_shards={num_shards})."
                )
        else:
            start_shard = 0
            start_position = 0
        self._shard_id = shard_id
        self._num_shards = num_shards

        for shard_index in range(start_shard, self.collection.sequence_count):
            shard_length = self.collection.shard_length(shard_index)
            local_count = (
                (shard_length - shard_id + num_shards - 1) // num_shards
                if shard_length > shard_id
                else 0
            )
            first_position = start_position if shard_index == start_shard else 0
            for position in range(first_position, local_count):
                self._current_shard = shard_index
                self._current_position = position + 1
                token = (shard_index, shard_id + position * num_shards)
                item = self._decode_or_skip(token)
                if item is not None:
                    yield item
            self._current_shard = shard_index + 1
            self._current_position = 0


class _PackedFileCache:
    """
    Process-local least-recently-used cache of read-only file descriptors.

    The cache uses ``os.pread`` so concurrent reads do not mutate a shared file
    position. It discards inherited descriptors after ``fork()`` and is
    pickle-safe for data-loader workers.

    Args:
        max_open_files:
            Positive descriptor limit. The least recently used descriptor is
            closed when opening another source would exceed it.
    """

    def __init__(self, max_open_files: int = 32):
        if max_open_files < 1:
            raise ValueError("max_open_files must be positive")
        self.max_open_files = max_open_files
        self._pid = os.getpid()
        self._fds: OrderedDict[str, int] = OrderedDict()

    def read(self, path: str, start: int, end: int) -> bytes:
        """
        Read the exact half-open byte range ``[start, end)`` from ``path``.

        Raises:
            EOFError:
                If the underlying file returns fewer bytes than requested.
        """
        if is_valid_url(path):
            raise ValueError(
                "Packed lazy reads require local source files; "
                f"cannot use os.pread() with {path!r}"
            )
        if start < 0 or end < start:
            raise ValueError(f"Invalid packed byte range: [{start}, {end})")
        self._ensure_process()
        fd = self._fds.pop(path, None)
        if fd is None:
            fd = os.open(path, os.O_RDONLY)
        self._fds[path] = fd
        while len(self._fds) > self.max_open_files:
            _, evicted = self._fds.popitem(last=False)
            os.close(evicted)
        chunks = []
        position = start
        while position < end:
            chunk = os.pread(fd, end - position, position)
            if not chunk:
                received = position - start
                raise EOFError(
                    f"Short indexed read from {path}: requested [{start}, {end}), "
                    f"received {received} bytes"
                )
            chunks.append(chunk)
            position += len(chunk)
        return b"".join(chunks)

    def limit_to(self, max_open_files: int) -> None:
        """Tighten this shared cache's descriptor bound."""
        if max_open_files < 1:
            raise ValueError("max_open_files must be positive")
        self.max_open_files = min(self.max_open_files, max_open_files)
        while len(self._fds) > self.max_open_files:
            _, evicted = self._fds.popitem(last=False)
            os.close(evicted)

    def close(self) -> None:
        """Close all cached descriptors."""
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()

    def __getstate__(self):
        return {"max_open_files": self.max_open_files}

    def __setstate__(self, state):
        self.max_open_files = state["max_open_files"]
        self._pid = os.getpid()
        self._fds = OrderedDict()

    def __del__(self):
        if hasattr(self, "_fds"):
            self.close()

    def _ensure_process(self) -> None:
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()


def _file_cache_for_pack(
    index_pack: IndexPack, max_open_files: int
) -> _PackedFileCache:
    global _PACKED_FILE_CACHE_PID
    pid = os.getpid()
    if pid != _PACKED_FILE_CACHE_PID:
        _PACKED_FILE_CACHES.clear()
        _PACKED_FILE_CACHE_PID = pid
    cache = _PACKED_FILE_CACHES.get(index_pack)
    if cache is None:
        cache = _PackedFileCache(max_open_files)
        _PACKED_FILE_CACHES[index_pack] = cache
    else:
        cache.limit_to(max_open_files)
    return cache


_PACKED_FILE_CACHES: weakref.WeakKeyDictionary[
    IndexPack, _PackedFileCache
] = weakref.WeakKeyDictionary()
_PACKED_FILE_CACHE_PID = os.getpid()
