"""Stream bucket tokens without constructing the nested snapshot lists."""

import io
import pickle
import struct
from typing import Any

_MAGIC = b"LHOTSE-BUCKETS\x01"
_COUNT = struct.Struct("<I")


class _TokenPickler(pickle.Pickler):
    """Accept built-in token data without user-defined reconstruction functions."""

    def reducer_override(self, obj):
        """Custom token classes require the legacy state representation."""
        raise ValueError(
            f"Unsupported compact bucket token type: {type(obj).__name__}. "
            "Use compact_state=False for custom token objects."
        )


class _TokenUnpickler(pickle.Unpickler):
    """Do not execute reconstruction functions when decoding checkpoint tokens."""

    def find_class(self, module, name):
        """Reject globals even in an externally supplied token payload."""
        raise ValueError(f"Non-primitive compact bucket token: {module}.{name}.")


class _TokenWriter:
    """Write counts and existing tokens directly into one immutable snapshot."""

    def __init__(self):
        self.stream = io.BytesIO()
        self.stream.write(_MAGIC)
        # Bound memo retention to one bucket, not the whole snapshot.
        self.pickler = _TokenPickler(self.stream, protocol=4)

    def start_list(self, count: int) -> None:
        """Write a bucket/item/cut count without creating a corresponding list."""
        self.stream.write(_COUNT.pack(count))

    def write(self, value: Any) -> None:
        """Let the C-backed pickler traverse an existing graph token."""
        self.pickler.dump(value)

    def start_bucket(self, count: int) -> None:
        """Start a bucket with an independent pickle memo."""
        self.pickler.clear_memo()
        self.start_list(count)

    def finish(self) -> bytes:
        """Detach an immutable snapshot from the writer."""
        return self.stream.getvalue()


def pack_bucket_tokens(buckets: list) -> bytes:
    """Convert a legacy checkpoint; live capture uses the writer directly."""
    writer = _TokenWriter()
    writer.start_list(len(buckets))
    for bucket in buckets:
        writer.start_bucket(len(bucket))
        for item in bucket:
            writer.start_list(len(item))
            for token in item:
                writer.write(token)
    return writer.finish()


def unpack_bucket_tokens(data: bytes) -> list:
    """Decode bucket counts and primitive token records, rejecting malformed data."""
    if not isinstance(data, bytes) or not data.startswith(_MAGIC):
        raise ValueError("Invalid compact bucket token header or version.")
    stream = io.BytesIO(data)
    stream.seek(len(_MAGIC))

    def count():
        """Bound container counts by the supplied payload before allocating lists."""
        raw = stream.read(_COUNT.size)
        if len(raw) != _COUNT.size:
            raise ValueError("Truncated compact bucket token count.")
        value = _COUNT.unpack(raw)[0]
        if value > len(data) - stream.tell():
            raise ValueError("Invalid compact bucket token count.")
        return value

    def bucket():
        """Match the writer's independent memo for each bucket."""
        unpickler = _TokenUnpickler(stream)
        return [[unpickler.load() for _ in range(count())] for _ in range(count())]

    try:
        result = [bucket() for _ in range(count())]
    except (EOFError, pickle.UnpicklingError, OverflowError) as exc:
        raise ValueError("Malformed compact bucket token payload.") from exc
    if stream.tell() != len(data):
        raise ValueError("Trailing data in compact bucket tokens.")
    return result
