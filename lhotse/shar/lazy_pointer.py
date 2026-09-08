"""
Lazy pointer addressing for Lhotse Shar tar shards.

A "Shar pointer" is a string that uniquely identifies a single sample's data
member inside an indexed Shar tar shard, *without* requiring any tar header
read at construction time:

    <tar_path>?o=<offset>&e=<end_offset>[&n=<expected_member_name>[&s=1]]

where ``offset`` and ``end_offset`` delimit the sample's indexed tar byte
range. The optional percent-encoded member name enables load-time validation
and recovery when a manifest is filtered or reordered. Pointers without a
member name retain their original behavior and wire format. ``s=1`` makes name
validation strict, disabling the compatibility scan when the indexed range has
an unexpected name.

At load time, :func:`read_payload` parses the indexed tar headers (including PAX
and GNU long-name records) and reads the first regular member's payload. When an
expected name is present, it is checked as part of that parsing and does not
require a separate storage read.

The file extension is not encoded separately. Formats are sniffed from the
payload's magic bytes (audio: soundfile auto-detect from :class:`io.BytesIO`;
arrays: NPY ``\x93NUMPY`` vs lilcom).
"""

from __future__ import annotations

import os
import re
import tarfile
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Dict, Optional, Tuple
from urllib.parse import quote_from_bytes, unquote_to_bytes

from lhotse.audio.source import resolve_s3_to_local_mirror
from lhotse.audio.utils import AudioLoadingError
from lhotse.serialization import (
    AIStoreIOBackend,
    CompositeIOBackend,
    get_current_io_backend,
    open_best,
)
from lhotse.utils import Pathlike, is_valid_url

_POINTER_RE = re.compile(
    r"^(?P<tar>[^?]+)\?o=(?P<o>\d+)&e=(?P<e>\d+)"
    r"(?:&n=(?P<n>[^&]*)(?:&s=(?P<s>1))?)?$"
)
_MIRROR_ROOTS_ENV = "LHOTSE_S3_LOCAL_MIRROR_ROOTS"
_MAX_OPEN_FILES = 32
_MAX_RESOLVED_PATHS = 256


@dataclass
class _HandleEntry:
    handle: BinaryIO
    lock: threading.Lock
    users: int = 0
    member_index: Optional[Dict[str, Tuple[int, int]]] = None
    close_requested: bool = False


# Both caches are process-local. Handles are keyed by resolved physical path so
# logical S3 identities sharing a local mirror also share the same descriptor.
_HANDLES: OrderedDict[str, _HandleEntry] = OrderedDict()
_RESOLVED_PATHS: OrderedDict[Tuple[str, Optional[str]], str] = OrderedDict()
_REGISTRY_LOCK = threading.Lock()


def encode_pointer(
    tar_path: Pathlike,
    offset: int,
    end_offset: int,
    *,
    expected_name: Optional[str] = None,
    strict: bool = False,
) -> str:
    """Encode a Shar lazy-pointer string.

    ``expected_name`` is optional so existing callers retain the original wire
    format. When present, :func:`read_payload` validates it while loading the
    indexed range. By default, a mismatch falls back to a cached name index for
    compatibility with filtered manifests. Set ``strict=True`` when the byte
    range is authoritative: a mismatch then fails without scanning the tar.
    """
    if strict and expected_name is None:
        raise ValueError("strict Shar pointers require expected_name")
    pointer = f"{tar_path}?o={int(offset)}&e={int(end_offset)}"
    if expected_name is not None:
        encoded_name = quote_from_bytes(
            str(expected_name).encode("utf-8", errors="surrogateescape"), safe=""
        )
        pointer += f"&n={encoded_name}"
        if strict:
            pointer += "&s=1"
    return pointer


def decode_pointer(s: str) -> Tuple[str, int, int]:
    """Parse a Shar pointer into ``(tar_path, offset, end_offset)``.

    The return type intentionally remains a three-tuple when the pointer also
    carries an expected member name, preserving the public API.
    """
    tar_path, offset, end_offset, _expected_name, _strict = _decode_pointer(s)
    return tar_path, offset, end_offset


def decode_pointer_with_name(s: str) -> Tuple[str, int, int, Optional[str]]:
    """Parse a Shar pointer, including its optional expected member name."""
    tar_path, offset, end_offset, expected_name, _strict = _decode_pointer(s)
    return tar_path, offset, end_offset, expected_name


def is_shar_pointer(s: Any) -> bool:
    """Return True iff ``s`` is a string in Shar lazy-pointer form."""
    return isinstance(s, str) and _POINTER_RE.match(s) is not None


def read_payload(pointer: str) -> bytes:
    """Resolve a Shar lazy pointer to the underlying data member's payload."""
    tar_path, offset, end_offset, expected_name, strict = _decode_pointer(pointer)
    try:
        resolved_path, entry = _acquire_handle(tar_path)
        try:
            with entry.lock:
                if (
                    expected_name is not None
                    and not strict
                    and entry.member_index is not None
                ):
                    data = _read_named_payload(
                        entry.handle,
                        entry.member_index,
                        expected_name,
                        resolved_path,
                    )
                else:
                    data, actual_name = _read_first_regular_member(
                        entry.handle,
                        offset,
                        end_offset,
                        pointer,
                        expected_name=expected_name,
                    )
                    if expected_name is not None and actual_name != expected_name:
                        if strict:
                            raise AudioLoadingError(
                                f"Indexed tar member name mismatch for {pointer!r}: "
                                f"expected {expected_name!r}, found {actual_name!r}."
                            )
                        entry.member_index = _build_member_index(
                            entry.handle, resolved_path
                        )
                        data = _read_named_payload(
                            entry.handle,
                            entry.member_index,
                            expected_name,
                            resolved_path,
                        )
        finally:
            _release_handle(entry)
        if data is None:
            message = (
                f"Shar pointer {pointer!r} points at a placeholder "
                "(.nodata/.nometa) member."
            )
            if expected_name is None:
                raise RuntimeError(message)
            raise AudioLoadingError(message)
        return data
    except AudioLoadingError:
        raise
    except Exception as ex:
        if expected_name is None:
            raise
        raise AudioLoadingError(
            f"Failed to load indexed tar payload from Shar pointer {pointer!r}: {ex}"
        ) from ex


def close_all() -> None:
    """Close cached tar handles, deferring active ones until their read completes."""
    with _REGISTRY_LOCK:
        for path, entry in list(_HANDLES.items()):
            if entry.users:
                entry.close_requested = True
                continue
            _HANDLES.pop(path)
            try:
                entry.handle.close()
            except Exception:
                pass
        _RESOLVED_PATHS.clear()


def _decode_pointer(s: str) -> Tuple[str, int, int, Optional[str], bool]:
    match = _POINTER_RE.match(s)
    if match is None:
        raise ValueError(f"Not a Shar pointer: {s!r}")
    offset = int(match.group("o"))
    end_offset = int(match.group("e"))
    if end_offset < offset:
        raise ValueError(
            f"Invalid Shar pointer byte range [{offset}, {end_offset}): {s!r}"
        )
    encoded_name = match.group("n")
    expected_name = (
        unquote_to_bytes(encoded_name).decode("utf-8", errors="surrogateescape")
        if encoded_name is not None
        else None
    )
    return (
        match.group("tar"),
        offset,
        end_offset,
        expected_name,
        match.group("s") == "1",
    )


def resolve_pointer_path(tar_path: str) -> str:
    """Resolve and cache the physical path used to load a pointer's tar."""
    cache_key = (tar_path, os.environ.get(_MIRROR_ROOTS_ENV))
    with _REGISTRY_LOCK:
        try:
            resolved = _RESOLVED_PATHS.pop(cache_key)
        except KeyError:
            pass
        else:
            _RESOLVED_PATHS[cache_key] = resolved
            return resolved

    resolved = resolve_s3_to_local_mirror(tar_path)
    with _REGISTRY_LOCK:
        try:
            resolved = _RESOLVED_PATHS.pop(cache_key)
        except KeyError:
            pass
        _RESOLVED_PATHS[cache_key] = resolved
        while len(_RESOLVED_PATHS) > _MAX_RESOLVED_PATHS:
            _RESOLVED_PATHS.popitem(last=False)
    return resolved


def _acquire_handle(tar_path: str) -> Tuple[str, _HandleEntry]:
    resolved_path = resolve_pointer_path(tar_path)
    with _REGISTRY_LOCK:
        try:
            entry = _HANDLES.pop(resolved_path)
        except KeyError:
            pass
        else:
            entry.users += 1
            _HANDLES[resolved_path] = entry
            return resolved_path, entry

    new_entry = _HandleEntry(
        handle=_open_seekable(resolved_path),
        lock=threading.Lock(),
    )
    with _REGISTRY_LOCK:
        try:
            entry = _HANDLES.pop(resolved_path)
        except KeyError:
            entry = new_entry
        else:
            new_entry.handle.close()
        entry.users += 1
        _HANDLES[resolved_path] = entry
        _evict_unused_handles()
    return resolved_path, entry


def _open_seekable(path: str) -> BinaryIO:
    if path.startswith("ais://") or _uses_aistore_backend(path):
        from lhotse.ais import AISRangeReader

        return AISRangeReader(path)
    return open_best(path, "rb")


def _release_handle(entry: _HandleEntry) -> None:
    with _REGISTRY_LOCK:
        entry.users -= 1
        if entry.users == 0 and entry.close_requested:
            for path, candidate in list(_HANDLES.items()):
                if candidate is entry:
                    _HANDLES.pop(path)
                    entry.handle.close()
                    break
            return
        _evict_unused_handles()


def _evict_unused_handles() -> None:
    if len(_HANDLES) <= _MAX_OPEN_FILES:
        return
    for path, entry in list(_HANDLES.items()):
        if len(_HANDLES) <= _MAX_OPEN_FILES:
            break
        if entry.users:
            continue
        _HANDLES.pop(path)
        entry.handle.close()


def _read_exact_range(
    handle: BinaryIO, start: int, end: int, source_path: str
) -> bytes:
    handle.seek(start)
    data = handle.read(end - start)
    if len(data) != end - start:
        raise EOFError(
            f"Short Shar pointer read from {source_path}: requested "
            f"[{start}, {end}), received {len(data)} bytes"
        )
    return data


class _BoundedReader:
    """Seekable view over one byte range of an already-open file."""

    def __init__(self, handle: BinaryIO, start: int, end: int):
        self.handle = handle
        self.start = start
        self.end = end
        self.position = 0

    def read(self, size: int = -1) -> bytes:
        available = max(0, self.end - self.start - self.position)
        size = available if size is None or size < 0 else min(size, available)
        self.handle.seek(self.start + self.position)
        data = self.handle.read(size)
        self.position += len(data)
        return data

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            position = offset
        elif whence == os.SEEK_CUR:
            position = self.position + offset
        elif whence == os.SEEK_END:
            position = self.end - self.start + offset
        else:
            raise ValueError(f"Unsupported seek mode: {whence}")
        if position < 0:
            raise ValueError(f"Cannot seek before indexed tar range: {position}")
        self.position = position
        return position

    def tell(self) -> int:
        return self.position


def _read_first_regular_member(
    handle: BinaryIO,
    start: int,
    end: int,
    pointer: str,
    *,
    expected_name: Optional[str],
) -> Tuple[Optional[bytes], str]:
    if end <= start:
        raise EOFError(f"Empty indexed tar range [{start}, {end}) for {pointer!r}")
    bounded = _BoundedReader(handle, start, end)
    try:
        with tarfile.open(fileobj=bounded, mode="r:") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                if expected_name is not None and member.name != expected_name:
                    return None, member.name
                if Path(member.name).suffix in (".nodata", ".nometa"):
                    return None, member.name
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise RuntimeError(
                        f"Unable to extract tar member {member.name!r} for {pointer!r}."
                    )
                return extracted.read(), member.name
    except tarfile.TarError as ex:
        raise type(ex)(f"{ex} while reading Shar pointer {pointer!r}") from ex
    raise RuntimeError(f"Shar pointer {pointer!r} contains no regular tar member.")


def _build_member_index(
    handle: BinaryIO, source_path: str
) -> Dict[str, Tuple[int, int]]:
    index: Dict[str, Tuple[int, int]] = {}
    handle.seek(0)
    try:
        with tarfile.open(fileobj=handle, mode="r:") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                if member.name in index:
                    raise ValueError(
                        f"Duplicate tar member name {member.name!r} in {source_path}; "
                        "name-keyed lazy pointer access is ambiguous."
                    )
                index[member.name] = (member.offset_data, member.size)
    except tarfile.TarError as ex:
        raise type(ex)(f"{ex} while indexing tar members in {source_path}") from ex
    return index


def _read_named_payload(
    handle: BinaryIO,
    member_index: Dict[str, Tuple[int, int]],
    expected_name: str,
    source_path: str,
) -> Optional[bytes]:
    try:
        start, size = member_index[expected_name]
    except KeyError as ex:
        raise KeyError(
            f"Tar {source_path} has no member named {expected_name!r}."
        ) from ex
    if Path(expected_name).suffix in (".nodata", ".nometa"):
        return None
    return _read_exact_range(handle, start, start + size, source_path)


def _reset_after_fork() -> None:
    # Child processes inherit file descriptors and locks in unsafe states.
    # Do not close duplicated parent descriptors here; simply forget them.
    global _REGISTRY_LOCK
    _HANDLES.clear()
    _RESOLVED_PATHS.clear()
    _REGISTRY_LOCK = threading.Lock()


def _uses_aistore_backend(path: str) -> bool:
    """Return whether ``open_best`` would route this URL through AIStore."""
    if not is_valid_url(path):
        return False
    backend = get_current_io_backend()
    if isinstance(backend, CompositeIOBackend):
        for candidate in backend.backends:
            if candidate.handles_special_case(path):
                backend = candidate
                break
        else:
            for candidate in backend.backends:
                if candidate.is_applicable(path):
                    backend = candidate
                    break
    return isinstance(backend, AIStoreIOBackend)


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)
