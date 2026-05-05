"""
Lazy pointer addressing for Lhotse Shar tar shards.

A "Shar pointer" is a string that uniquely identifies a single sample's data
member inside an indexed Shar tar shard, *without* requiring any tar header
read at construction time:

    <tar_path>?o=<offset>&e=<end_offset>

where ``offset`` is the byte offset of the sample's data tar header (sourced
from the existing ``.idx`` sidecar's offset array) and ``end_offset`` is the
offset of the next sample's data header (or the tar's total file size for the
last sample, taken from the ``.idx`` sentinel). The half-open interval
``[offset, end_offset)`` covers ``data_header(512) + data_payload + pad +
meta_header(512) + meta_payload + pad``.

At load time, :func:`read_payload` does a single ranged read of that interval,
parses the leading 512 bytes via :func:`lhotse.indexing.read_tar_member_at`,
and returns just the payload bytes.

The pointer never encodes the file extension or member name — formats are
sniffed from the payload's magic bytes (audio: soundfile auto-detect from
:class:`io.BytesIO`; arrays: NPY ``\x93NUMPY`` vs lilcom).
"""

from __future__ import annotations

import os
import re
import threading
from typing import Any, Dict, Tuple

from lhotse.indexing import read_tar_member_at
from lhotse.serialization import open_best
from lhotse.utils import Pathlike

_POINTER_RE = re.compile(r"^(?P<tar>[^?]+)\?o=(?P<o>\d+)&e=(?P<e>\d+)$")

# Process-local file-handle reuse for repeated payload fetches from the same
# tar. Intentionally not an LRU — typical workloads have tens of shards in
# flight; eviction is unnecessary. ``close_all()`` is exposed for tests.
#
# Each tar gets its own ``threading.Lock`` so concurrent readers on different
# tars don't serialize against each other; the global ``_REGISTRY_LOCK``
# only guards lookup/insertion in the registry itself.
_HANDLES: Dict[str, Tuple[Any, threading.Lock]] = {}
_REGISTRY_LOCK = threading.Lock()


def encode_pointer(tar_path: Pathlike, offset: int, end_offset: int) -> str:
    """Encode a Shar lazy-pointer string."""
    return f"{tar_path}?o={int(offset)}&e={int(end_offset)}"


def decode_pointer(s: str) -> Tuple[str, int, int]:
    """Parse a Shar lazy-pointer string into ``(tar_path, offset, end_offset)``."""
    m = _POINTER_RE.match(s)
    if m is None:
        raise ValueError(f"Not a Shar pointer: {s!r}")
    return m.group("tar"), int(m.group("o")), int(m.group("e"))


def is_shar_pointer(s: Any) -> bool:
    """Return True iff ``s`` is a string in Shar lazy-pointer form."""
    return isinstance(s, str) and _POINTER_RE.match(s) is not None


def _get_handle(tar_path: str) -> Tuple[Any, threading.Lock]:
    with _REGISTRY_LOCK:
        entry = _HANDLES.get(tar_path)
        if entry is None:
            entry = (open_best(tar_path, "rb"), threading.Lock())
            _HANDLES[tar_path] = entry
        return entry


def read_payload(pointer: str) -> bytes:
    """Resolve a Shar lazy pointer to the underlying data member's payload."""
    tar_path, offset, _end_offset = decode_pointer(pointer)
    fh, fh_lock = _get_handle(tar_path)
    with fh_lock:
        data, _path, _info = read_tar_member_at(fh, offset)
    if data is None:
        raise RuntimeError(
            f"Shar pointer {pointer!r} points at a placeholder (.nodata/.nometa) member."
        )
    return data


def close_all() -> None:
    """Close all cached tar file handles. Intended for tests / cleanup."""
    with _REGISTRY_LOCK:
        for fh, _lock in _HANDLES.values():
            try:
                fh.close()
            except Exception:
                pass
        _HANDLES.clear()


# Worker processes inherit ``_HANDLES`` across ``fork()`` but inherit only
# duplicated FDs from the parent — concurrent reads from parent + child against
# the same FD will corrupt each other's seek positions. Reset the registry in
# the child so each worker opens its own fresh handle on first use.
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=lambda: _HANDLES.clear())
