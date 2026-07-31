"""
Packed, memory-mapped random-access indexes for sharded byte-addressable data.

An ``.idxpack`` combines the contents of many conventional little-endian
``uint64`` ``.idx`` sidecars into one immutable file. Its catalog and offset
payload are accessed through a single read-only mmap, so opening a large
collection does not require one filesystem operation or one in-memory offset
array per source shard.

This module is deliberately independent of manifest schemas and downstream
frameworks. Callers describe each logical collection with an application-defined
role, an arbitrary storage-kind string, the original source specification, and
the ordered concrete source paths. The pack stores that catalog and provides
byte-range lookup; interpreting the bytes remains the caller's responsibility.
"""

from __future__ import annotations

import hashlib
import json
import mmap
import os
import struct
import uuid
import weakref
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from lhotse.indexing import index_file_path
from lhotse.utils import is_valid_url

# Keep these values stable: production packs created by the original format
# implementation already use this signature and version.
_MAGIC = b"IDXPACK2"
_VERSION = 2
_HEADER_SIZE = 256

# Header fields:
# magic, format version, header size,
# (offset, count/size) for collections, sequences, segments, strings, offsets,
# layout SHA-256.
_HEADER = struct.Struct("<8sIIQQQQQQQQQQ32s")

# Collection fields:
# key, sequence start, sequence count, total records,
# kind-string position, kind-string length, reserved flags.
_COLLECTION = struct.Struct("<32sQQQQII")
_COLLECTION_PATHS_ONLY = 1

# Sequence fields:
# segment ID, cumulative record count through this shard.
_SEQUENCE = struct.Struct("<QQ")

# Segment fields:
# path-string position, offsets position, path-string length, flags,
# offset count, source size, offsets byte size, CRC32, reserved.
_SEGMENT = struct.Struct("<QQIIQQQII")
_SEGMENT_PATH_ONLY = 1
_U64 = struct.Struct("<Q")


def index_pack_collection_key(role: str, kind: str, source_spec) -> bytes:
    """
    Return the stable SHA-256 key for one logical collection.

    Args:
        role:
            Application-defined purpose of the collection, such as
            ``"manifest"`` or ``"features"``. The value is not interpreted by
            Lhotse.
        kind:
            Application-defined storage/serialization identifier. Any non-empty
            string is accepted and persisted in the pack.
        source_spec:
            JSON-serializable source description used by the caller before
            concrete paths were expanded. It participates in identity so two
            differently declared collections may reference the same paths.

    Returns:
        A 32-byte digest used to retrieve the collection from an
        :class:`IndexPack`.
    """
    _validate_collection_identity(role, kind)
    payload = json.dumps(
        {
            "kind": kind,
            "role": role,
            "source_spec": _canonicalize(source_spec),
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).digest()


@dataclass(frozen=True)
class IndexPackCollectionSpec:
    """
    Build-time description of one ordered logical collection.

    Args:
        role:
            Application-defined purpose of this collection. It distinguishes
            collections that use the same storage kind and source declaration
            for different purposes; Lhotse does not interpret it.
        kind:
            Application-defined storage/serialization name, for example
            ``"json-lines"`` or ``"tar-members"``. The string is persisted for
            runtime validation but does not change how offsets are stored.
        source_spec:
            JSON-serializable source expression before expansion (a path
            template, list, or mapping, for example). Together with ``role`` and
            ``kind`` it defines :attr:`key`; it is not otherwise interpreted.
        paths:
            Concrete source paths in logical shard order. By default every path
            must have a conventional ``.idx`` sidecar whose final uint64 is the
            source-size sentinel.
        offsets_required:
            When true, copy and validate each path's sidecar. When false, retain
            only the ordered paths; the resulting collection has zero logical
            records and is intended for callers that only need
            :meth:`PackedIndexCollection.path_for_shard`.
    """

    role: str
    kind: str
    source_spec: object
    paths: tuple[str, ...]
    offsets_required: bool = True

    def __post_init__(self):
        _validate_collection_identity(self.role, self.kind)
        object.__setattr__(self, "paths", tuple(str(path) for path in self.paths))

    @property
    def key(self) -> bytes:
        """Return the stable lookup key derived from this specification."""
        return index_pack_collection_key(self.role, self.kind, self.source_spec)


@dataclass(frozen=True)
class PackedIndexLocation:
    """
    Resolved source byte range for one logical record.

    Attributes:
        path:
            Concrete source path containing the record.
        start:
            Inclusive byte offset in ``path``.
        end:
            Exclusive byte offset in ``path``.
        segment_id:
            Physical segment-table row used for the lookup. A segment may be
            shared by several logical collections.
        shard_index:
            Position of the source path in this collection's ordered shard
            sequence.
        local_index:
            Zero-based record position within that shard.
    """

    path: str
    start: int
    end: int
    segment_id: int
    shard_index: int
    local_index: int


def write_index_pack(
    output_path,
    collections: Sequence[IndexPackCollectionSpec],
    *,
    indexes_root=None,
    overwrite: bool = False,
) -> Path:
    """
    Convert existing sidecar indexes into one atomic ``.idxpack``.

    Collection order and path order are preserved. Repeated physical sources
    are stored once inside a pack and referenced by multiple sequence entries.
    The output is written to a temporary sibling, flushed with ``fsync()``, and
    atomically published after all sidecars pass structural validation.

    Args:
        output_path:
            Destination pack path.
        collections:
            Logical collections belonging to one dataset. A dataset may contain
            several collections (for example records and payload members).
        indexes_root:
            Optional mirror root passed to :func:`lhotse.indexing.index_file_path`
            when resolving each conventional sidecar.
        overwrite:
            Replace an existing destination atomically. The default is to fail
            if ``output_path`` already exists.

    Returns:
        ``output_path`` normalized to :class:`~pathlib.Path`.

    Raises:
        FileNotFoundError:
            If a required sidecar or local source is absent.
        ValueError:
            If identities collide, a source is newer than its sidecar, offsets
            are malformed/non-monotonic, or a sentinel disagrees with source
            size.

    Example:
        Build one pack per dataset for a two-dataset composition. Each pack may
        still contain multiple logical collections::

            datasets = {
                "books": [
                    IndexPackCollectionSpec(
                        role="records",
                        kind="json-lines",
                        source_spec="books-{000..127}.jsonl",
                        paths=tuple(book_shards),
                    )
                ],
                "speech": [
                    IndexPackCollectionSpec(
                        role="records",
                        kind="json-lines",
                        source_spec="speech-{000..255}.jsonl",
                        paths=tuple(speech_manifests),
                    ),
                    IndexPackCollectionSpec(
                        role="payload",
                        kind="tar-members",
                        source_spec="speech-{000..255}.tar",
                        paths=tuple(speech_tars),
                    ),
                ],
            }
            for dataset_name, specs in datasets.items():
                write_index_pack(
                    f"{dataset_name}.idxpack",
                    specs,
                    indexes_root="/srv/index-mirror",
                )
    """
    output_path = Path(output_path)
    collections = tuple(collections)
    if not collections:
        raise ValueError("Cannot build an index pack without collections.")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Index pack already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    collection_keys: set[bytes] = set()
    segments: list[_BuildSegment] = []
    segment_ids: dict[tuple[str, bool], int] = {}
    sequences: list[tuple[int, int]] = []
    collection_rows: list[tuple[bytes, int, int, int, int, int, int]] = []
    strings = _StringTableBuilder()

    for collection in collections:
        if collection.key in collection_keys:
            raise ValueError(
                "Duplicate collection key in index pack. Distinguish repeated logical "
                f"collections with a different role/source spec: {collection.source_spec!r}"
            )
        collection_keys.add(collection.key)
        sequence_start = len(sequences)
        cumulative_end = 0
        for path in collection.paths:
            segment_key = (path, collection.offsets_required)
            segment_id = segment_ids.get(segment_key)
            if segment_id is None:
                segment_id = len(segments)
                segment_ids[segment_key] = segment_id
                segments.append(
                    _read_sidecar_metadata(
                        path,
                        indexes_root,
                        offsets_required=collection.offsets_required,
                    )
                )
            cumulative_end += segments[segment_id].num_records
            sequences.append((segment_id, cumulative_end))
        kind_position, kind_length = strings.add(collection.kind)
        collection_rows.append(
            (
                collection.key,
                sequence_start,
                len(collection.paths),
                cumulative_end,
                kind_position,
                kind_length,
                0 if collection.offsets_required else _COLLECTION_PATHS_ONLY,
            )
        )

    path_positions = [strings.add(segment.path) for segment in segments]
    string_blob = bytes(strings.data)

    collection_offset = _HEADER_SIZE
    sequence_offset = collection_offset + len(collection_rows) * _COLLECTION.size
    segment_offset = sequence_offset + len(sequences) * _SEQUENCE.size
    strings_offset = segment_offset + len(segments) * _SEGMENT.size
    offsets_offset = strings_offset + len(string_blob)
    offsets_offset += (-offsets_offset) % _U64.size
    offsets_size = sum(segment.offsets_count * _U64.size for segment in segments)
    layout_hash = _layout_digest(collections)

    tmp_path = output_path.with_name(
        f".{output_path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    segment_rows = []
    try:
        with tmp_path.open("w+b") as out:
            header = _HEADER.pack(
                _MAGIC,
                _VERSION,
                _HEADER_SIZE,
                collection_offset,
                len(collection_rows),
                sequence_offset,
                len(sequences),
                segment_offset,
                len(segments),
                strings_offset,
                len(string_blob),
                offsets_offset,
                offsets_size,
                layout_hash,
            )
            out.write(header)
            out.write(b"\0" * (_HEADER_SIZE - len(header)))

            for (
                key,
                sequence_start,
                sequence_count,
                total_records,
                kind_rel,
                kind_len,
                flags,
            ) in collection_rows:
                out.write(
                    _COLLECTION.pack(
                        key,
                        sequence_start,
                        sequence_count,
                        total_records,
                        strings_offset + kind_rel,
                        kind_len,
                        flags,
                    )
                )
            for row in sequences:
                out.write(_SEQUENCE.pack(*row))

            # Filled after payload copy, once each sidecar CRC is known.
            out.write(b"\0" * (len(segments) * _SEGMENT.size))

            out.write(string_blob)
            if out.tell() < offsets_offset:
                out.write(b"\0" * (offsets_offset - out.tell()))

            payload_cursor = offsets_offset
            for segment_id, segment in enumerate(segments):
                expected_size = segment.offsets_count * _U64.size
                checksum = 0
                copied = 0
                previous: int | None = None
                if segment.path_only:
                    chunk = _U64.pack(0)
                    checksum = zlib.crc32(chunk)
                    copied = len(chunk)
                    previous = 0
                    out.write(chunk)
                else:
                    assert segment.index_path is not None
                    with segment.index_path.open("rb") as src:
                        while chunk := src.read(1024 * 1024):
                            if len(chunk) % _U64.size:
                                raise ValueError(
                                    f"Index chunk is not uint64-aligned: {segment.index_path}"
                                )
                            for (value,) in struct.iter_unpack("<Q", chunk):
                                if previous is not None and value < previous:
                                    raise ValueError(
                                        f"Non-monotonic offsets in {segment.index_path}: "
                                        f"{value} follows {previous}"
                                    )
                                previous = value
                            checksum = zlib.crc32(chunk, checksum)
                            copied += len(chunk)
                            out.write(chunk)
                if copied != expected_size:
                    raise ValueError(
                        f"Index changed while packing {segment.index_path}: "
                        f"expected {expected_size} bytes, copied {copied}"
                    )
                if previous is None:
                    raise ValueError(
                        f"Index sidecar contains no sentinel: {segment.index_path}"
                    )
                source_size = (
                    previous if segment.source_size is None else segment.source_size
                )
                if previous != source_size:
                    raise ValueError(
                        f"Invalid sentinel in {segment.index_path}: metadata={source_size}, payload={previous}"
                    )
                path_rel, path_len = path_positions[segment_id]
                segment_rows.append(
                    (
                        strings_offset + path_rel,
                        payload_cursor,
                        path_len,
                        _SEGMENT_PATH_ONLY if segment.path_only else 0,
                        segment.offsets_count,
                        source_size,
                        expected_size,
                        checksum & 0xFFFFFFFF,
                        0,
                    )
                )
                payload_cursor += expected_size

            if out.tell() != offsets_offset + offsets_size:
                raise AssertionError(
                    f"Internal idxpack size mismatch: {out.tell()} != {offsets_offset + offsets_size}"
                )
            out.seek(segment_offset)
            for row in segment_rows:
                out.write(_SEGMENT.pack(*row))
            out.flush()
            os.fsync(out.fileno())
        if overwrite:
            os.replace(tmp_path, output_path)
        else:
            try:
                os.link(tmp_path, output_path)
            except FileExistsError as ex:
                raise FileExistsError(
                    f"Index pack already exists: {output_path}"
                ) from ex
            else:
                tmp_path.unlink()
        _fsync_directory(output_path.parent)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return output_path


class PackedIndexCollection:
    """
    Zero-copy view of one logical collection inside an :class:`IndexPack`.

    Instances are created by :meth:`IndexPack.collection`; callers normally do
    not instantiate this class directly. It translates a collection-global
    record index into a source path and byte range without materializing the
    shard catalog or offsets.

    Args:
        pack:
            Owning memory-mapped pack.
        key:
            Stable 32-byte collection identity.
        sequence_start:
            First row in the pack-wide shard-sequence table.
        sequence_count:
            Number of ordered shards in this collection.
        total_records:
            Sum of records across the collection's indexed shards.
        kind:
            Caller-defined storage/serialization string persisted by the
            corresponding :class:`IndexPackCollectionSpec`.
        offsets_required:
            Whether this collection contains record offsets. Path-only
            collections support :meth:`path_for_shard` but contain no records.
    """

    def __init__(
        self,
        pack: IndexPack,
        key: bytes,
        sequence_start: int,
        sequence_count: int,
        total_records: int,
        kind: str,
        offsets_required: bool,
    ):
        self.pack = pack
        self.key = key
        self.sequence_start = sequence_start
        self.sequence_count = sequence_count
        self.total_records = total_records
        self.kind = kind
        self.offsets_required = offsets_required

    def __len__(self) -> int:
        """Return the number of addressable records across all shards."""
        return self.total_records

    def path_for_shard(self, shard_index: int) -> str:
        """
        Return a concrete source path by logical shard position.

        Args:
            shard_index:
                Zero-based shard position. Negative indices follow Python
                sequence semantics.

        Returns:
            The persisted source path.

        Raises:
            IndexError:
                If ``shard_index`` is outside the collection.
        """
        if shard_index < 0:
            shard_index += self.sequence_count
        if shard_index < 0 or shard_index >= self.sequence_count:
            raise IndexError(
                f"shard index {shard_index} out of range for packed collection "
                f"with {self.sequence_count} shards"
            )
        self.pack._ensure_open()
        segment_id, _ = self.pack._sequence(self.sequence_start + shard_index)
        segment = self.pack._segment(segment_id)
        path_position, _, path_length = segment[:3]
        return self.pack._string(
            path_position, path_length, label=f"segment {segment_id} path"
        )

    def shard_length(self, shard_index: int) -> int:
        """
        Return the number of records in one logical shard.

        Negative shard indices follow Python sequence semantics.
        """
        shard_index = self._normalize_shard_index(shard_index)
        _, cumulative_end = self.pack._sequence(self.sequence_start + shard_index)
        previous_end = (
            self.pack._sequence(self.sequence_start + shard_index - 1)[1]
            if shard_index
            else 0
        )
        return cumulative_end - previous_end

    def locate_in_shard(
        self, shard_index: int, local_index: int
    ) -> PackedIndexLocation:
        """
        Resolve a shard-local record index to its source byte range.

        Both indices accept Python-style negative values.
        """
        shard_index = self._normalize_shard_index(shard_index)
        shard_length = self.shard_length(shard_index)
        if local_index < 0:
            local_index += shard_length
        if local_index < 0 or local_index >= shard_length:
            raise IndexError(
                f"local index {local_index} out of range for packed shard "
                f"{shard_index} with {shard_length} records"
            )
        pack = self.pack
        pack._ensure_open()
        segment_id, _ = pack._sequence(self.sequence_start + shard_index)
        segment = pack._segment(segment_id)
        offsets_position = segment[1]
        start = pack._u64(offsets_position + local_index * _U64.size)
        end = pack._u64(offsets_position + (local_index + 1) * _U64.size)
        if end < start or end > segment[5]:
            raise ValueError(
                f"Corrupt idxpack offsets for segment {segment_id}: [{start}, {end}) "
                f"outside source size {segment[5]}"
            )
        path_position, _, path_length = segment[:3]
        return PackedIndexLocation(
            path=pack._string(
                path_position, path_length, label=f"segment {segment_id} path"
            ),
            start=start,
            end=end,
            segment_id=segment_id,
            shard_index=shard_index,
            local_index=local_index,
        )

    def locate(self, index: int) -> PackedIndexLocation:
        """
        Resolve a collection-global record index to its source byte range.

        A binary search over cumulative shard lengths finds the physical
        segment, after which two uint64 values are read directly from the mmap.

        Args:
            index:
                Collection-global record index. Negative indices follow Python
                sequence semantics.

        Returns:
            The source path, byte range, and resolved shard/local positions.

        Raises:
            IndexError:
                If ``index`` is outside the collection.
            ValueError:
                If sequence metadata or offsets are internally inconsistent.
        """
        if index < 0:
            index += self.total_records
        if index < 0 or index >= self.total_records:
            raise IndexError(
                f"index {index} out of range for packed collection with {self.total_records} records"
            )
        pack = self.pack
        pack._ensure_open()
        lo = 0
        hi = self.sequence_count
        while lo < hi:
            mid = (lo + hi) // 2
            _, cumulative_end = pack._sequence(self.sequence_start + mid)
            if cumulative_end <= index:
                lo = mid + 1
            else:
                hi = mid
        shard_index = lo
        if shard_index >= self.sequence_count:
            raise ValueError(
                "Corrupt idxpack collection: record index exceeds the final cumulative shard count"
            )
        previous_end = (
            pack._sequence(self.sequence_start + shard_index - 1)[1]
            if shard_index
            else 0
        )
        return self.locate_in_shard(shard_index, index - previous_end)

    def _normalize_shard_index(self, shard_index: int) -> int:
        if shard_index < 0:
            shard_index += self.sequence_count
        if shard_index < 0 or shard_index >= self.sequence_count:
            raise IndexError(
                f"shard index {shard_index} out of range for packed collection "
                f"with {self.sequence_count} shards"
            )
        return shard_index


class IndexPack:
    """
    Lazy, read-only view of an ``.idxpack``.

    Construction reads the compact collection catalog with temporary
    positional reads and closes the file before returning. This makes lengths,
    kinds, and collection lookup available while a data-loader graph is being
    assembled without leaving a file descriptor or mmap to be inherited by
    worker processes. The complete pack is opened, deeply validated, and
    memory-mapped only when a record, shard path, or segment is first accessed.
    Offset payloads are never copied into Python or NumPy memory.

    The lazy mapping is process-local and the object is pickle-safe for
    data-loader workers. Use it as a context manager when a deterministic close
    is desired, or use :func:`open_index_pack` to share one view per absolute
    path and process.

    Args:
        path:
            Local path to an ``.idxpack`` file. The pack itself must be
            seekable/mappable even when its indexed source paths are remote.
        expected_layout_hash:
            Optional 32-byte digest or hexadecimal string to compare with the
            pack header. This pins the exact collection identities and ordered
            source paths expected by the caller.

    Attributes:
        path:
            Normalized :class:`~pathlib.Path` of the mapped pack.
        layout_hash:
            32-byte digest of collection identities and ordered paths.
        num_collections:
            Number of logical collections.
        num_sequences:
            Number of collection-to-segment sequence rows.
        num_segments:
            Number of deduplicated physical source/index segments.

    Example::

        spec = IndexPackCollectionSpec(
            role="records",
            kind="json-lines",
            source_spec="dataset-{000..127}.jsonl",
            paths=tuple(expanded_paths),
        )
        write_index_pack("dataset.idxpack", [spec])
        with IndexPack("dataset.idxpack") as pack:
            location = pack.collection(spec.key).locate(10)
            with open(location.path, "rb") as source:
                source.seek(location.start)
                record = source.read(location.end - location.start)
    """

    def __init__(self, path, *, expected_layout_hash: str | bytes | None = None):
        self.path = Path(path)
        self.expected_layout_hash = expected_layout_hash
        self._fh = None
        self._mmap = None
        self._pid = None
        self._file_identity = None
        self._collections = {}
        self._read_catalog()

    def collection(self, key: bytes | str) -> PackedIndexCollection:
        """
        Return a zero-copy logical collection view.

        Args:
            key:
                The 32-byte result of :func:`index_pack_collection_key`, or its
                64-character hexadecimal representation.

        Returns:
            A view supporting length, shard-path, and record-location lookup.

        Raises:
            KeyError:
                If no collection with ``key`` exists in this pack.
        """
        if isinstance(key, str):
            key = bytes.fromhex(key)
        try:
            (
                sequence_start,
                sequence_count,
                total_records,
                kind,
                offsets_required,
            ) = self._collections[key]
        except KeyError as ex:
            raise KeyError(
                f"Collection {key.hex()} is not present in index pack {self.path}"
            ) from ex
        return PackedIndexCollection(
            self,
            key,
            sequence_start,
            sequence_count,
            total_records,
            kind,
            offsets_required,
        )

    def verify_segment(self, segment_id: int) -> None:
        """
        Verify one packed offset payload against its stored CRC32.

        Validation is explicit because scanning every offset payload at open
        time would defeat fast startup.

        Args:
            segment_id:
                Zero-based physical segment-table row.

        Raises:
            IndexError:
                If ``segment_id`` is outside the segment table.
            ValueError:
                If the payload checksum does not match.
        """
        self._ensure_open()
        segment = self._segment(segment_id)
        offsets_position = segment[1]
        offsets_size = segment[6]
        expected_crc = segment[7]
        actual_crc = (
            zlib.crc32(self._mmap[offsets_position : offsets_position + offsets_size])
            & 0xFFFFFFFF
        )
        if actual_crc != expected_crc:
            raise ValueError(
                f"Index-pack CRC mismatch for segment {segment_id} in {self.path}: "
                f"expected={expected_crc:#x}, actual={actual_crc:#x}"
            )

    def close(self) -> None:
        """Close the mmap and its underlying file descriptor."""
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        self._pid = None

    def __enter__(self):
        """Return this mapped pack for context-manager use."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Close the mapping when leaving a context manager."""
        self.close()

    def __del__(self):
        if hasattr(self, "_mmap"):
            self.close()

    def __getstate__(self):
        return {
            "path": self.path,
            "expected_layout_hash": self.expected_layout_hash,
            "file_identity": self._file_identity,
            "catalog": {
                "collection_offset": self.collection_offset,
                "num_collections": self.num_collections,
                "sequence_offset": self.sequence_offset,
                "num_sequences": self.num_sequences,
                "segment_offset": self.segment_offset,
                "num_segments": self.num_segments,
                "strings_offset": self.strings_offset,
                "strings_size": self.strings_size,
                "offsets_offset": self.offsets_offset,
                "offsets_size": self.offsets_size,
                "layout_hash": self.layout_hash,
                "collections": self._collections,
            },
        }

    def __setstate__(self, state):
        self.path = state["path"]
        self.expected_layout_hash = state["expected_layout_hash"]
        self._fh = None
        self._mmap = None
        self._pid = None
        self._file_identity = state.get("file_identity")
        catalog = state.get("catalog")
        if catalog is None:
            # Compatibility with objects pickled before catalog-only opening.
            self._collections = {}
            self._read_catalog()
            return
        self.collection_offset = catalog["collection_offset"]
        self.num_collections = catalog["num_collections"]
        self.sequence_offset = catalog["sequence_offset"]
        self.num_sequences = catalog["num_sequences"]
        self.segment_offset = catalog["segment_offset"]
        self.num_segments = catalog["num_segments"]
        self.strings_offset = catalog["strings_offset"]
        self.strings_size = catalog["strings_size"]
        self.offsets_offset = catalog["offsets_offset"]
        self.offsets_size = catalog["offsets_size"]
        self.layout_hash = catalog["layout_hash"]
        self._collections = catalog["collections"]

    def _read_header(self, source, file_size: int) -> None:
        """Decode and validate the common on-disk section layout."""
        (
            magic,
            version,
            header_size,
            self.collection_offset,
            self.num_collections,
            self.sequence_offset,
            self.num_sequences,
            self.segment_offset,
            self.num_segments,
            self.strings_offset,
            self.strings_size,
            self.offsets_offset,
            self.offsets_size,
            self.layout_hash,
        ) = _HEADER.unpack_from(source, 0)
        if magic != _MAGIC:
            raise ValueError(
                f"Invalid index-pack header magic in {self.path}: {magic!r}"
            )
        if version != _VERSION or header_size != _HEADER_SIZE:
            raise ValueError(
                f"Unsupported index-pack header in {self.path}: "
                f"version={version}, header_size={header_size}"
            )
        sections = (
            (
                "collections",
                self.collection_offset,
                self.num_collections * _COLLECTION.size,
            ),
            (
                "sequences",
                self.sequence_offset,
                self.num_sequences * _SEQUENCE.size,
            ),
            (
                "segments",
                self.segment_offset,
                self.num_segments * _SEGMENT.size,
            ),
            ("strings", self.strings_offset, self.strings_size),
            ("offsets", self.offsets_offset, self.offsets_size),
        )
        for name, offset, size in sections:
            if offset < _HEADER_SIZE or size < 0 or offset + size > file_size:
                raise ValueError(
                    f"Index pack has truncated/invalid {name} section: "
                    f"offset={offset}, size={size}, file_size={file_size}"
                )
        expected_sections = (
            ("collections", self.collection_offset, _HEADER_SIZE),
            (
                "sequences",
                self.sequence_offset,
                self.collection_offset + self.num_collections * _COLLECTION.size,
            ),
            (
                "segments",
                self.segment_offset,
                self.sequence_offset + self.num_sequences * _SEQUENCE.size,
            ),
            (
                "strings",
                self.strings_offset,
                self.segment_offset + self.num_segments * _SEGMENT.size,
            ),
        )
        for name, actual, expected_offset in expected_sections:
            if actual != expected_offset:
                raise ValueError(
                    f"Index pack has invalid {name} offset: "
                    f"{actual} != {expected_offset}"
                )
        expected_offsets_offset = self.strings_offset + self.strings_size
        expected_offsets_offset += (-expected_offsets_offset) % _U64.size
        if (
            self.offsets_offset != expected_offsets_offset
            or self.offsets_offset + self.offsets_size != file_size
        ):
            raise ValueError(
                "Index pack sections overlap, contain gaps, or do not cover the complete file"
            )
        expected = self.expected_layout_hash
        if expected is not None:
            if isinstance(expected, str):
                expected = bytes.fromhex(expected)
            if expected != self.layout_hash:
                raise ValueError(
                    f"Index-pack layout mismatch for {self.path}: "
                    f"expected={expected.hex()}, actual={self.layout_hash.hex()}"
                )

    def _read_catalog(self) -> None:
        """
        Read construction-time metadata without retaining an fd or mmap.

        Deep validation of every sequence and segment is intentionally deferred
        to :meth:`_open`. The catalog phase validates the file layout,
        collection directory, collection endpoints, and the first segment used
        to determine whether each collection is path-only.
        """
        try:
            fh = self.path.open("rb")
        except FileNotFoundError as ex:
            raise FileNotFoundError(f"Index pack not found: {self.path}") from ex
        try:
            stat = os.fstat(fh.fileno())
            identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
            if self._file_identity is not None and identity != self._file_identity:
                raise RuntimeError(
                    f"Index pack changed after it was opened: {self.path}; "
                    "reconstruct the dataset to use the replacement"
                )
            file_size = stat.st_size
            if file_size < _HEADER_SIZE:
                raise ValueError(
                    f"Index pack is truncated before its {_HEADER_SIZE}-byte header: {self.path}"
                )
            header = _pread_exact(fh.fileno(), _HEADER_SIZE, 0)
            self._read_header(header, file_size)

            collections = {}
            expected_sequence_start = 0
            collection_table = _pread_exact(
                fh.fileno(),
                self.num_collections * _COLLECTION.size,
                self.collection_offset,
            )
            for collection_id in range(self.num_collections):
                (
                    key,
                    sequence_start,
                    sequence_count,
                    total_records,
                    kind_position,
                    kind_length,
                    flags,
                ) = _COLLECTION.unpack_from(
                    collection_table, collection_id * _COLLECTION.size
                )
                if flags & ~_COLLECTION_PATHS_ONLY:
                    raise ValueError(
                        f"Index pack collection {collection_id} has unsupported flags: {flags:#x}"
                    )
                if (
                    sequence_start != expected_sequence_start
                    or sequence_start + sequence_count > self.num_sequences
                ):
                    raise ValueError(
                        f"Index pack collection {collection_id} has an invalid sequence range"
                    )
                if key in collections:
                    raise ValueError(
                        f"Duplicate collection key in index pack: {key.hex()}"
                    )
                kind = self._pread_string(
                    fh.fileno(),
                    kind_position,
                    kind_length,
                    label=f"collection {collection_id} kind",
                )
                declared_paths_only = bool(flags & _COLLECTION_PATHS_ONLY)
                paths_only = declared_paths_only
                if sequence_count:
                    segment_id, _ = _SEQUENCE.unpack(
                        _pread_exact(
                            fh.fileno(),
                            _SEQUENCE.size,
                            self.sequence_offset + sequence_start * _SEQUENCE.size,
                        )
                    )
                    if segment_id >= self.num_segments:
                        raise ValueError(
                            f"Index pack collection {collection_id} has corrupt sequence metadata"
                        )
                    segment = _SEGMENT.unpack(
                        _pread_exact(
                            fh.fileno(),
                            _SEGMENT.size,
                            self.segment_offset + segment_id * _SEGMENT.size,
                        )
                    )
                    paths_only = bool(segment[3] & _SEGMENT_PATH_ONLY)
                    _, final_cumulative = _SEQUENCE.unpack(
                        _pread_exact(
                            fh.fileno(),
                            _SEQUENCE.size,
                            self.sequence_offset
                            + (sequence_start + sequence_count - 1) * _SEQUENCE.size,
                        )
                    )
                    if final_cumulative != total_records:
                        raise ValueError(
                            f"Index pack collection {collection_id} has corrupt "
                            f"cumulative count for its final shard: "
                            f"{final_cumulative} != {total_records}"
                        )
                if declared_paths_only and not paths_only:
                    raise ValueError(
                        f"Index pack collection {collection_id} is marked path-only "
                        "but references indexed segments"
                    )
                if paths_only and total_records != 0:
                    raise ValueError(
                        f"Index pack collection {collection_id} has an invalid total record count"
                    )
                collections[key] = (
                    sequence_start,
                    sequence_count,
                    total_records,
                    kind,
                    not paths_only,
                )
                expected_sequence_start += sequence_count
            if expected_sequence_start != self.num_sequences:
                raise ValueError("Index pack contains unreferenced sequence rows")
            self._collections = collections
            self._file_identity = identity
        finally:
            fh.close()

    def _pread_string(self, fd: int, position: int, length: int, *, label: str) -> str:
        if (
            position < self.strings_offset
            or position + length > self.strings_offset + self.strings_size
        ):
            raise ValueError(
                f"Index pack {label} points outside the strings section: "
                f"position={position}, length={length}"
            )
        try:
            return _pread_exact(fd, length, position).decode("utf-8")
        except UnicodeDecodeError as ex:
            raise ValueError(f"Index pack {label} is not valid UTF-8") from ex

    def _open(self) -> None:
        try:
            self._fh = self.path.open("rb")
        except FileNotFoundError as ex:
            raise FileNotFoundError(f"Index pack not found: {self.path}") from ex
        stat = os.fstat(self._fh.fileno())
        identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if self._file_identity is not None and identity != self._file_identity:
            self._fh.close()
            self._fh = None
            raise RuntimeError(
                f"Index pack changed after it was opened: {self.path}; "
                "reconstruct the dataset to use the replacement"
            )
        file_size = stat.st_size
        if file_size < _HEADER_SIZE:
            self._fh.close()
            self._fh = None
            raise ValueError(
                f"Index pack is truncated before its {_HEADER_SIZE}-byte header: {self.path}"
            )
        self._mmap = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        self._pid = os.getpid()
        self._file_identity = identity
        try:
            self._read_header(self._mmap, file_size)
        except Exception:
            self.close()
            raise
        offsets_cursor = self.offsets_offset
        for segment_id in range(self.num_segments):
            segment = self._segment(segment_id)
            (
                path_position,
                offsets_position,
                path_length,
                flags,
                offsets_count,
                source_size,
                size,
                _,
                _,
            ) = segment
            if flags & ~_SEGMENT_PATH_ONLY:
                self.close()
                raise ValueError(
                    f"Index pack segment {segment_id} has unsupported flags: {flags:#x}"
                )
            self._string(path_position, path_length, label=f"segment {segment_id} path")
            if offsets_count < 1 or size != offsets_count * _U64.size:
                self.close()
                raise ValueError(
                    f"Index pack segment {segment_id} has inconsistent offset count/size"
                )
            if (
                offsets_position != offsets_cursor
                or offsets_position + size > self.offsets_offset + self.offsets_size
            ):
                self.close()
                raise ValueError(
                    f"Index pack segment {segment_id} has an invalid offset payload range"
                )
            if flags & _SEGMENT_PATH_ONLY and (offsets_count != 1 or source_size != 0):
                self.close()
                raise ValueError(
                    f"Index pack path-only segment {segment_id} contains record metadata"
                )
            offsets_cursor += size
        if offsets_cursor != self.offsets_offset + self.offsets_size:
            self.close()
            raise ValueError(
                "Index pack segment payloads do not cover the offsets section"
            )

        collections = {}
        expected_sequence_start = 0
        for collection_id in range(self.num_collections):
            row = _COLLECTION.unpack_from(
                self._mmap,
                self.collection_offset + collection_id * _COLLECTION.size,
            )
            (
                key,
                sequence_start,
                sequence_count,
                total_records,
                kind_position,
                kind_length,
                flags,
            ) = row
            if flags & ~_COLLECTION_PATHS_ONLY:
                self.close()
                raise ValueError(
                    f"Index pack collection {collection_id} has unsupported flags: {flags:#x}"
                )
            if (
                sequence_start != expected_sequence_start
                or sequence_start + sequence_count > self.num_sequences
            ):
                self.close()
                raise ValueError(
                    f"Index pack collection {collection_id} has an invalid sequence range"
                )
            if key in collections:
                self.close()
                raise ValueError(f"Duplicate collection key in index pack: {key.hex()}")
            kind = self._string(
                kind_position, kind_length, label=f"collection {collection_id} kind"
            )
            cumulative = 0
            declared_paths_only = bool(flags & _COLLECTION_PATHS_ONLY)
            paths_only = None
            for local_shard in range(sequence_count):
                segment_id, cumulative_end = self._sequence(
                    sequence_start + local_shard
                )
                if segment_id >= self.num_segments:
                    self.close()
                    raise ValueError(
                        f"Index pack collection {collection_id} has corrupt sequence metadata"
                    )
                segment = self._segment(segment_id)
                expected_cumulative_end = cumulative + segment[4] - 1
                if cumulative_end != expected_cumulative_end:
                    self.close()
                    raise ValueError(
                        f"Index pack collection {collection_id} has corrupt "
                        f"cumulative count for shard {local_shard}: "
                        f"{cumulative_end} != {expected_cumulative_end}"
                    )
                segment_paths_only = bool(segment[3] & _SEGMENT_PATH_ONLY)
                if paths_only is None:
                    paths_only = segment_paths_only
                elif segment_paths_only != paths_only:
                    self.close()
                    raise ValueError(
                        f"Index pack collection {collection_id} mixes path-only and indexed segments"
                    )
                cumulative = cumulative_end
            if paths_only is None:
                paths_only = declared_paths_only
            if declared_paths_only and not paths_only:
                self.close()
                raise ValueError(
                    f"Index pack collection {collection_id} is marked path-only "
                    "but references indexed segments"
                )
            if cumulative != total_records or (paths_only and total_records != 0):
                self.close()
                raise ValueError(
                    f"Index pack collection {collection_id} has an invalid total record count"
                )
            collections[key] = (
                sequence_start,
                sequence_count,
                total_records,
                kind,
                not paths_only,
            )
            expected_sequence_start += sequence_count

        if expected_sequence_start != self.num_sequences:
            self.close()
            raise ValueError("Index pack contains unreferenced sequence rows")
        self._collections = collections

    def _ensure_open(self) -> None:
        if self._mmap is None or self._pid != os.getpid():
            self.close()
            self._open()
            _register_index_pack(self)

    def _sequence(self, index: int) -> tuple[int, int]:
        self._ensure_open()
        if index < 0 or index >= self.num_sequences:
            raise IndexError(f"Index-pack sequence index out of range: {index}")
        return _SEQUENCE.unpack_from(
            self._mmap, self.sequence_offset + index * _SEQUENCE.size
        )

    def _segment(self, index: int):
        self._ensure_open()
        if index < 0 or index >= self.num_segments:
            raise IndexError(f"Index-pack segment index out of range: {index}")
        return _SEGMENT.unpack_from(
            self._mmap, self.segment_offset + index * _SEGMENT.size
        )

    def _u64(self, position: int) -> int:
        self._ensure_open()
        return _U64.unpack_from(self._mmap, position)[0]

    def _string(self, position: int, length: int, *, label: str) -> str:
        self._ensure_open()
        if (
            position < self.strings_offset
            or position + length > self.strings_offset + self.strings_size
        ):
            raise ValueError(
                f"Index pack {label} points outside the strings section: position={position}, length={length}"
            )
        try:
            return self._mmap[position : position + length].decode("utf-8")
        except UnicodeDecodeError as ex:
            raise ValueError(f"Index pack {label} is not valid UTF-8") from ex


def open_index_pack(path) -> IndexPack:
    """
    Return one shared lazy pack view per absolute path and process.

    Args:
        path:
            Local ``.idxpack`` path.

    Returns:
        A process-local cached :class:`IndexPack`. Construction retains only
        the catalog; each child creates its own mapping on first data access.
    """
    global _INDEX_PACK_CACHE_PID
    pid = os.getpid()
    if pid != _INDEX_PACK_CACHE_PID:
        _INDEX_PACK_CACHE.clear()
        _INDEX_PACK_CACHE_PID = pid
    key = str(Path(path).absolute())
    pack = _INDEX_PACK_CACHE.get(key)
    if pack is None:
        pack = IndexPack(key)
        _INDEX_PACK_CACHE[key] = pack
    return pack


@dataclass(frozen=True)
class _BuildSegment:
    """
    Normalized metadata for one physical source/index pair while writing.

    Attributes:
        path:
            Persisted source path.
        index_path:
            Resolved local sidecar path, or ``None`` for a path-only segment.
        offsets_count:
            Number of uint64 values, including the final source-size sentinel.
            Therefore the addressable record count is one less.
        source_size:
            Current local source size when it can be checked, otherwise
            ``None`` and the copied sidecar sentinel becomes authoritative.
        path_only:
            Whether this segment intentionally stores no record offsets.
    """

    path: str
    index_path: Path | None
    offsets_count: int
    source_size: int | None
    path_only: bool = False

    @property
    def num_records(self) -> int:
        """Return addressable records represented by this segment."""
        return self.offsets_count - 1


class _StringTableBuilder:
    """Deduplicating UTF-8 string table used while writing a pack."""

    def __init__(self):
        self.data = bytearray()
        self._positions: dict[bytes, tuple[int, int]] = {}

    def add(self, value: str) -> tuple[int, int]:
        encoded = value.encode("utf-8")
        position = self._positions.get(encoded)
        if position is None:
            position = (len(self.data), len(encoded))
            self._positions[encoded] = position
            self.data.extend(encoded)
        return position


def _validate_collection_identity(role: str, kind: str) -> None:
    if not isinstance(role, str) or not role:
        raise ValueError(f"Index-pack role must be a non-empty string, got {role!r}")
    if not isinstance(kind, str) or not kind:
        raise ValueError(f"Index-pack kind must be a non-empty string, got {kind!r}")


def _canonicalize(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _canonicalize(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonicalize(item) for item in value]
    return value


def _read_sidecar_metadata(
    path: str, indexes_root, *, offsets_required: bool
) -> _BuildSegment:
    if not offsets_required:
        return _BuildSegment(
            path=path,
            index_path=None,
            offsets_count=1,
            source_size=0,
            path_only=True,
        )
    idx = index_file_path(path, indexes_root)
    if _is_remote_path(idx):
        raise ValueError(
            "Index-pack conversion currently requires a local sidecar; "
            f"got remote index path: {idx}"
        )
    idx = Path(idx)
    try:
        index_stat = idx.stat()
    except FileNotFoundError as ex:
        raise FileNotFoundError(f"Missing .idx sidecar for {path}: {idx}") from ex
    size = index_stat.st_size
    if size < _U64.size or size % _U64.size:
        raise ValueError(
            f"Invalid .idx sidecar {idx}: size must be a positive multiple of {_U64.size}, got {size}"
        )

    source_size = None
    if not _is_remote_path(path):
        try:
            source_stat = Path(path).stat()
        except FileNotFoundError as ex:
            raise FileNotFoundError(f"Indexed source not found: {path}") from ex
        if source_stat.st_mtime_ns > index_stat.st_mtime_ns:
            raise ValueError(
                f"Source {path} is newer than index sidecar {idx}; rebuild the .idx before packing"
            )
        source_size = source_stat.st_size
    return _BuildSegment(
        path=path,
        index_path=idx,
        offsets_count=size // _U64.size,
        source_size=source_size,
    )


def _layout_digest(collections: Sequence[IndexPackCollectionSpec]) -> bytes:
    digest = hashlib.sha256()
    for collection in collections:
        digest.update(collection.key)
        digest.update(bytes((collection.offsets_required,)))
        digest.update(_U64.pack(len(collection.paths)))
        for path in collection.paths:
            encoded = path.encode("utf-8")
            digest.update(_U64.pack(len(encoded)))
            digest.update(encoded)
    return digest.digest()


def _is_remote_path(path) -> bool:
    return is_valid_url(str(path))


def _pread_exact(fd: int, size: int, offset: int) -> bytes:
    """Read exactly ``size`` bytes at ``offset`` without changing fd position."""
    chunks = []
    remaining = size
    while remaining:
        chunk = os.pread(fd, remaining, offset)
        if not chunk:
            raise EOFError(
                f"Short positional read: requested {size} bytes at offset "
                f"{offset - (size - remaining)}, received {size - remaining}"
            )
        chunks.append(chunk)
        offset += len(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _fsync_directory(path: Path) -> None:
    if not hasattr(os, "O_DIRECTORY"):
        return
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _register_index_pack(pack: IndexPack) -> None:
    global _INDEX_PACK_CACHE_PID
    pid = os.getpid()
    if pid != _INDEX_PACK_CACHE_PID:
        _INDEX_PACK_CACHE.clear()
        _INDEX_PACK_CACHE_PID = pid
    _INDEX_PACK_CACHE[str(pack.path.absolute())] = pack


_INDEX_PACK_CACHE: weakref.WeakValueDictionary[
    str, IndexPack
] = weakref.WeakValueDictionary()
_INDEX_PACK_CACHE_PID = os.getpid()
