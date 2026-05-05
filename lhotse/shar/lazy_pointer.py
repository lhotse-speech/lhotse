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
parses the leading 512 bytes as a :class:`tarfile.TarInfo` to learn the
payload size, and returns just the payload bytes.

The pointer never encodes the file extension or member name — formats are
sniffed from the payload's magic bytes (audio: soundfile auto-detect from
:class:`io.BytesIO`; arrays: NPY ``\x93NUMPY`` vs lilcom).
"""

from __future__ import annotations

import re
import tarfile
import threading
from typing import Any, Dict, Tuple

from lhotse.serialization import open_best
from lhotse.utils import Pathlike

_TAR_BLOCK_SIZE = 512

_POINTER_RE = re.compile(r"^(?P<tar>[^?]+)\?o=(?P<o>\d+)&e=(?P<e>\d+)$")

# Process-local file-handle reuse for repeated payload fetches from the same
# tar. Intentionally not an LRU — typical workloads have tens of shards in
# flight; eviction is unnecessary. ``close_all()`` is exposed for tests.
_HANDLES: Dict[str, Any] = {}
_LOCK = threading.Lock()


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


def _get_handle(tar_path: str):
    with _LOCK:
        fh = _HANDLES.get(tar_path)
        if fh is None:
            fh = open_best(tar_path, "rb")
            _HANDLES[tar_path] = fh
        return fh


def read_payload(pointer: str) -> bytes:
    """
    Resolve a Shar lazy pointer to the underlying data member's payload bytes.

    Performs a single ranged read of ``[offset, end_offset)``, then parses the
    leading 512 bytes as a :class:`tarfile.TarInfo` to extract the payload
    portion.
    """
    tar_path, offset, end_offset = decode_pointer(pointer)
    length = end_offset - offset
    if length <= _TAR_BLOCK_SIZE:
        raise RuntimeError(
            f"Shar pointer {pointer!r} has window size {length} <= tar block "
            f"size {_TAR_BLOCK_SIZE}; index is likely corrupted."
        )
    fh = _get_handle(tar_path)
    with _LOCK:
        fh.seek(offset)
        block = fh.read(length)
    if len(block) < _TAR_BLOCK_SIZE:
        raise RuntimeError(
            f"Short read for Shar pointer {pointer!r}: expected at least "
            f"{_TAR_BLOCK_SIZE} bytes, got {len(block)}."
        )
    info = tarfile.TarInfo.frombuf(
        block[:_TAR_BLOCK_SIZE], tarfile.ENCODING, "surrogateescape"
    )
    payload_end = _TAR_BLOCK_SIZE + info.size
    if payload_end > len(block):
        raise RuntimeError(
            f"Shar pointer {pointer!r} window is too small to contain the "
            f"declared payload of {info.size} bytes."
        )
    return block[_TAR_BLOCK_SIZE:payload_end]


def close_all() -> None:
    """Close all cached tar file handles. Intended for tests / cleanup."""
    with _LOCK:
        for fh in _HANDLES.values():
            try:
                fh.close()
            except Exception:
                pass
        _HANDLES.clear()
