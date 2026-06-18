"""
Seekable file-like wrapper over AIStore HTTP byte-range reads.

The default :class:`lhotse.serialization.AIStoreIOBackend.open` path returns a
non-seekable ``ObjectFile`` stream (great for sequential reads, fatal for the
indexed readers in :mod:`lhotse.indexing` which jump to arbitrary tar/jsonl
offsets). :class:`AISRangeReader` solves this by translating ``seek()`` +
``read(n)`` into ``Object.get_reader(byte_range=…)`` HTTP range requests
served by AIStore in O(1) per call.

Reused by:

* :class:`lhotse.indexing.IndexedJsonlReader` and
  :class:`lhotse.indexing.IndexedTarMemberReader` (via
  :func:`lhotse.indexing._open_for_indexed_read`).
* ``NeMo`` ``nemo.collections.common.data.lhotse.indexed_adapters`` (it imports
  :class:`AISRangeReader` from here so the two libraries don't drift).
"""

from typing import Optional


class AISRangeReader:
    """
    Pseudo file-like object backed by AIStore HTTP byte-range reads.

    Translates ``seek()`` + ``read(n)`` into ``Object.get_reader(byte_range=…)``
    requests so the indexed-tar / indexed-jsonl readers can do random access
    into ``s3://`` / ``ais://`` archives the same way they would into a local
    file. Each ``read()`` is one HTTP range request, which AIStore serves in
    O(1); the index already tells us exactly which byte ranges we need.

    The aistore SDK is imported lazily through
    :func:`lhotse.serialization.get_aistore_client` so this module remains
    importable on pure-local installs.

    Notes
    -----
    * ``seek()`` accepts ``whence ∈ {0, 1, 2}``; for ``whence=2`` the size
      already cached from ``Object.props.size`` is used, so ``seek(0, 2)``
      does not issue an extra HTTP call.
    * Pickling drops the cached ``_obj`` so forked workers re-resolve the
      URL on first access in the child process.
    * Thread-safe across reads ONLY in the sense that each ``read()`` is one
      HTTP call with its own AIStore reader. Sharing a single instance
      across threads will interleave ``_pos`` updates — use one per worker.
    """

    def __init__(self, url: str):
        self._url = url
        self._obj = None
        self._size: Optional[int] = None
        self._pos = 0

    def _ensure_obj(self):
        if self._obj is not None:
            return
        # Same client/env wiring as ``lhotse.serialization.AIStoreIOBackend`` —
        # imported locally so build_indexes / non-remote callers don't pull
        # the aistore SDK in.
        from lhotse.serialization import get_aistore_client

        client, _version = get_aistore_client()
        self._obj = client.get_object_from_url(self._url)
        self._size = int(self._obj.props.size)

    @property
    def size(self) -> int:
        self._ensure_obj()
        return self._size  # type: ignore[return-value]

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = int(offset)
        elif whence == 1:
            self._pos += int(offset)
        elif whence == 2:
            self._pos = self.size + int(offset)
        else:
            raise ValueError(f"Unsupported whence: {whence}")
        return self._pos

    def tell(self) -> int:
        return self._pos

    def read(self, n: int = -1) -> bytes:
        self._ensure_obj()
        if self._pos >= self._size:
            return b""
        if n == 0:
            return b""
        if n < 0:
            end_inclusive = self._size - 1
        else:
            end_inclusive = min(self._pos + n - 1, self._size - 1)
        if end_inclusive < self._pos:
            return b""
        # AIStore expects the HTTP Range syntax ``bytes=START-END`` with END
        # INCLUSIVE. ``read_all()`` drains the entire response into bytes.
        byte_range = f"bytes={self._pos}-{end_inclusive}"
        reader = self._obj.get_reader(byte_range=byte_range)
        data = reader.read_all()
        self._pos += len(data)
        return data

    def close(self) -> None:
        self._obj = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_obj"] = None  # AIS client/object not picklable across forks
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
