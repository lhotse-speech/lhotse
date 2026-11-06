"""
Index packs (``.idxpack``) for Lhotse Shar.

A conventional indexed Shar reader opens one ``.idx`` sidecar per shard and per field when it is
constructed. For a large dataset that means hundreds of thousands of small files. An index pack stores all of
those offset arrays in a single memory-mapped file (see :mod:`lhotse.index_pack`), so
:class:`~lhotse.shar.readers.indexed.LazyIndexedSharIterator` can be constructed from one file and reads the
shards through a bounded pool of file descriptors.

Use :func:`write_shar_index_pack` to build a pack from existing sidecars (``lhotse index shar`` creates them),
and pass the result to ``CutSet.from_shar(..., index_pack=...)`` or ``LazyIndexedSharIterator(..., index_pack=...)``.

**Source identity.** A pack may hold several Shar *sources*. Every source is identified by the arguments that a
reader is given, so a reader finds its source in the pack by passing the same arguments:

* a local Shar directory or an AIStore/S3 location, identified by its path or URL
  (use the same spelling when writing and reading; a trailing slash does not matter);
* a per-field mapping ``{"cuts": [shard, ...], "<field>": [shard, ...], ...}``, identified by the mapping's
  content: the field names and the ordered shard paths as strings.

A pack written from a directory can therefore not be read through a mapping of the same files (and vice versa);
pack both forms as two sources if both are needed. Files shared by several sources are stored once.

**Layout.** For every source the pack holds a ``shar_cuts`` collection (one record per cut), a path-only
``shar_fields`` collection that lists the field names, and one ``shar_field`` collection per field with one record
per cut (a tar member pair for tar fields, a JSONL row for JSONL fields).
"""

import io
import os
import shutil
import struct
import tempfile
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Generator, List, Optional, Sequence, Tuple, Union

from lhotse.index_pack import (
    IndexPack,
    IndexPackCollectionSpec,
    index_pack_collection_key,
    open_index_pack,
    write_index_pack,
)
from lhotse.indexing import (
    _is_gzip_jsonl,
    _open_for_indexed_read,
    gzip_index_file_path,
    index_file_path,
    indexed_path_kind,
    read_tar_pair_at,
)
from lhotse.packed_lazy import read_packed_range
from lhotse.serialization import decode_json_line, open_best
from lhotse.utils import Pathlike, is_valid_url

SHAR_CUTS_ROLE = "shar_cuts"
SHAR_FIELDS_ROLE = "shar_fields"
SHAR_FIELD_ROLE = "shar_field"
_CUTS_KIND = "jsonl"
_FIELDS_KIND = "shar_fields"
_FIELD_KINDS = ("tar", "jsonl")

_U64 = struct.Struct("<Q")

ShardSource = Union[Pathlike, Mapping]


def write_shar_index_pack(
    output_path: Pathlike,
    sources: Union[ShardSource, Sequence[ShardSource]],
    *,
    indexes_root: Optional[Pathlike] = None,
    overwrite: bool = False,
) -> Path:
    """
    Pack the ``.idx`` sidecars of one or more Shar sources into a single ``.idxpack``.

    Each source is either a Shar directory, or a per-field mapping
    ``{"cuts": [<shard>, ...], "<field>": [<shard>, ...], ...}`` like the ``fields`` argument of
    :class:`~lhotse.shar.readers.indexed.LazyIndexedSharIterator`.
    A reader is later pointed at the pack and at its source with the same ``in_dir`` / ``fields`` arguments; see the
    module docstring for how a source is identified. One pack can serve several readers, e.g. one reader per
    directory of a multiplexed dataset.

    Create the sidecars first with ``lhotse index shar`` or ``SharWriter(create_index=True)``.
    Shards and sidecars may be local files or AIStore/S3 objects. Remote sidecars
    are downloaded while building the pack. The finished pack is a local file.

    Each field must have the same number of shards and records as the cuts.
    Lhotse checks shard sizes and tar padding before writing the pack. For local
    files, it also checks that the sidecars are newer than the shards.
    Gzip JSONL requires its ``.gzidx`` seek index; tar shards must be uncompressed.

    :param output_path: where to write the pack.
    :param sources: one Shar source, or a sequence of them (see above).
    :param indexes_root: optional root that mirrors the data layout for sidecar lookup, as in
        :func:`lhotse.indexing.index_file_path`.
    :param overwrite: replace an existing pack atomically instead of failing.
    :return: ``output_path`` as a :class:`~pathlib.Path`.
    """
    if isinstance(sources, (str, os.PathLike, Mapping)):
        sources = [sources]
    resolved = [_resolve_source(source) for source in sources]
    if not resolved:
        raise ValueError("Cannot build an index pack without Shar sources.")

    specs = []
    seen = set()
    for source_spec, streams in resolved:
        key = shar_cuts_collection_key(source_spec)
        if key in seen:
            raise ValueError(f"Shar source {source_spec} is listed more than once.")
        seen.add(key)
        specs.extend(_collection_specs(source_spec, streams))

    shard_paths = sorted(
        {
            path
            for _, streams in resolved
            for paths in streams.values()
            for path in paths
        }
    )
    with _local_sidecars(paths=shard_paths, indexes_root=indexes_root) as index_paths:
        record_counts = {}
        for _, streams in resolved:
            _check_record_counts(streams, index_paths, record_counts)
        source_size_overrides = _source_sizes(
            paths=shard_paths, index_paths=index_paths
        )
        return write_index_pack(
            output_path,
            specs,
            indexes_root=indexes_root,
            overwrite=overwrite,
            source_size_overrides=source_size_overrides,
            index_path_overrides=index_paths,
        )


def shar_source_spec(
    fields: Optional[Mapping] = None, in_dir: Optional[Pathlike] = None
) -> dict:
    """
    Return the identity of a Shar source (a JSON-serializable dict) for the given reader arguments.
    See the module docstring for the rule.
    """
    if (fields is None) == (in_dir is None):
        raise ValueError("Provide exactly one of 'fields' and 'in_dir'.")
    if in_dir is not None:
        return {
            "in_dir": (
                str(in_dir).rstrip("/") if is_valid_url(in_dir) else str(Path(in_dir))
            )
        }
    if "cuts" not in fields:
        raise ValueError("A Shar 'fields' mapping must contain the key 'cuts'.")
    return {"fields": {str(k): [str(p) for p in v] for k, v in fields.items()}}


def shar_cuts_collection_key(source_spec: dict) -> bytes:
    return index_pack_collection_key(SHAR_CUTS_ROLE, _CUTS_KIND, source_spec)


def shar_fields_collection_key(source_spec: dict) -> bytes:
    return index_pack_collection_key(SHAR_FIELDS_ROLE, _FIELDS_KIND, source_spec)


def shar_field_collection_key(source_spec: dict, field: str, kind: str) -> bytes:
    return index_pack_collection_key(
        SHAR_FIELD_ROLE, kind, {"source": source_spec, "field": field}
    )


class PackedSharSource:
    """
    One Shar source (see the module docstring) inside an :class:`~lhotse.index_pack.IndexPack`.

    It resolves global cut indexes and hands out per-shard readers that mirror the interface of
    :class:`~lhotse.indexing.IndexedJsonlReader` and :class:`~lhotse.indexing.IndexedTarReader`.
    No sidecar file is opened. The object is cheap to pickle: only the pack handle and the source identity are kept.

    :param index_pack: an :class:`IndexPack` or the path to a ``.idxpack`` file.
    :param source_spec: the source identity from :func:`shar_source_spec`.
    :param max_open_files: bound of the descriptor LRU shared by all readers of this pack in a process.
    """

    def __init__(
        self,
        index_pack: Union[IndexPack, Pathlike],
        source_spec: dict,
        *,
        max_open_files: int = 32,
    ) -> None:
        if max_open_files < 1:
            raise ValueError("index_pack_max_open_files must be positive.")
        self.index_pack = (
            index_pack
            if isinstance(index_pack, IndexPack)
            else open_index_pack(index_pack)
        )
        self.source_spec = source_spec
        self.max_open_files = max_open_files
        self._bind()

    def _bind(self) -> None:
        pack = self.index_pack
        try:
            self._cuts = pack.collection(shar_cuts_collection_key(self.source_spec))
            names = pack.collection(shar_fields_collection_key(self.source_spec))
        except KeyError as ex:
            raise ValueError(
                f"Index pack {pack.path} has no Shar source {self.source_spec}. "
                "A source is found by the same 'in_dir' / 'fields' arguments it was packed with; "
                "see lhotse.shar.write_shar_index_pack."
            ) from ex
        self.field_names = [
            names.path_for_shard(i) for i in range(names.sequence_count)
        ]
        self._fields = {name: self._find_field(name) for name in self.field_names}
        for name, (collection, _) in self._fields.items():
            if collection.sequence_count != self._cuts.sequence_count or len(
                collection
            ) != len(self._cuts):
                raise ValueError(
                    f"Packed Shar field '{name}' is inconsistent with the cuts in {pack.path}: "
                    f"{collection.sequence_count} shards / {len(collection)} records vs "
                    f"{self._cuts.sequence_count} / {len(self._cuts)}."
                )

    def _find_field(self, name: str):
        for kind in _FIELD_KINDS:
            try:
                key = shar_field_collection_key(self.source_spec, name, kind)
                return self.index_pack.collection(key), kind
            except KeyError:
                continue
        raise ValueError(
            f"Index pack {self.index_pack.path} lists field '{name}' of {self.source_spec} but does not hold it."
        )

    @property
    def num_shards(self) -> int:
        return self._cuts.sequence_count

    def __len__(self) -> int:
        return len(self._cuts)

    def locate(self, idx: int) -> Tuple[int, int]:
        """Map a global cut index (negative values allowed) to ``(shard_idx, position_within_shard)``."""
        location = self._cuts.locate(idx)
        return location.shard_index, location.local_index

    def cuts_shard_path(self, shard_idx: int) -> str:
        return self._cuts.path_for_shard(shard_idx)

    def cuts_reader(self, shard_idx: int) -> "PackedJsonlShardReader":
        return PackedJsonlShardReader(
            self.index_pack, self._cuts, shard_idx, self.max_open_files
        )

    def field_reader(
        self, field: str, shard_idx: int
    ) -> Union["PackedTarShardReader", "PackedJsonlShardReader"]:
        collection, kind = self._fields[field]
        reader_cls = PackedTarShardReader if kind == "tar" else PackedJsonlShardReader
        return reader_cls(self.index_pack, collection, shard_idx, self.max_open_files)

    def __getstate__(self):
        return {
            "index_pack": self.index_pack,
            "source_spec": self.source_spec,
            "max_open_files": self.max_open_files,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._bind()


class PackedJsonlShardReader:
    """Random access to the rows of one JSONL shard through an index pack."""

    def __init__(self, index_pack, collection, shard_idx: int, max_open_files: int):
        self._pack = index_pack
        self._collection = collection
        self._shard_idx = shard_idx
        self._max_open_files = max_open_files

    @property
    def path(self) -> str:
        return self._collection.path_for_shard(self._shard_idx)

    def __len__(self) -> int:
        return self._collection.shard_length(self._shard_idx)

    def _read(self, idx: int) -> bytes:
        location = self._collection.locate_in_shard(self._shard_idx, idx)
        return read_packed_range(
            self._pack,
            location.path,
            location.start,
            location.end,
            max_open_files=self._max_open_files,
        )

    def __getitem__(self, idx: int) -> dict:
        return decode_json_line(self._read(idx).decode("utf-8"))


class PackedTarShardReader(PackedJsonlShardReader):
    """Random access to the (data, metadata) member pairs of one Shar tar shard through an index pack."""

    def __getitem__(self, idx: int):
        return read_tar_pair_at(io.BytesIO(self._read(idx)))

    def read_metadata(self, idx: int):
        # The pair is fetched as one byte range; only its payload is not decoded.
        return read_tar_pair_at(io.BytesIO(self._read(idx)), read_data=False)

    def member_byte_range(self, idx: int) -> Tuple[int, int]:
        location = self._collection.locate_in_shard(self._shard_idx, idx)
        return location.start, location.end


def _resolve_source(source: ShardSource) -> Tuple[dict, Dict[str, List[str]]]:
    """Return ``(source identity, {"cuts": [...], field: [...]})`` of a directory or a mapping, validated."""
    if isinstance(source, Mapping):
        for field, shards in source.items():
            if isinstance(shards, (str, os.PathLike)) or not isinstance(
                shards, Sequence
            ):
                raise ValueError(
                    f"Shar field '{field}' must map to a list of shards, got {shards!r}."
                )
        source_spec = shar_source_spec(fields=source)
        streams = {k: list(v) for k, v in source_spec["fields"].items()}
    else:
        from lhotse.shar.readers.discovery import discover_shar_fields

        source_spec = shar_source_spec(in_dir=source)
        try:
            _, found = discover_shar_fields(in_dir=source)
        except AssertionError as ex:
            raise ValueError(f"No Shar cuts shards found in {source}.") from ex
        streams = {k: [str(p) for p in v] for k, v in found.items()}
    if not streams["cuts"]:
        raise ValueError(f"No cuts shards in Shar source {source_spec}.")
    for field, shards in streams.items():
        for path in shards:
            _check_shard(field, path)
        if len(shards) != len(streams["cuts"]):
            raise ValueError(
                f"Shar field '{field}' has {len(shards)} shards but there are {len(streams['cuts'])} cuts shards "
                f"in {source_spec}."
            )
    return source_spec, streams


def _check_shard(field: str, path: str) -> None:
    if path.startswith("pipe:"):
        raise ValueError(
            f"Index packs require seekable shards, got '{path}' (field '{field}')."
        )
    kind = indexed_path_kind(path)
    if field == "cuts" and kind != "jsonl":
        raise ValueError(f"Cuts shards must be JSONL files, got '{path}'.")
    if kind is None:
        raise ValueError(
            f"Shard '{path}' of field '{field}' is compressed or not a tar/JSONL file; "
            "such shards are not randomly addressable and cannot be packed."
        )


def _collection_specs(
    source_spec: dict, streams: Dict[str, List[str]]
) -> List[IndexPackCollectionSpec]:
    fields = sorted(f for f in streams if f != "cuts")
    specs = [
        IndexPackCollectionSpec(
            role=SHAR_CUTS_ROLE,
            kind=_CUTS_KIND,
            source_spec=source_spec,
            paths=tuple(streams["cuts"]),
        ),
        IndexPackCollectionSpec(
            role=SHAR_FIELDS_ROLE,
            kind=_FIELDS_KIND,
            source_spec=source_spec,
            paths=tuple(fields),
            offsets_required=False,
        ),
    ]
    for field in fields:
        kind = indexed_path_kind(streams[field][0])
        if any(indexed_path_kind(p) != kind for p in streams[field]):
            raise ValueError(f"Shar field '{field}' mixes tar and JSONL shards.")
        specs.append(
            IndexPackCollectionSpec(
                role=SHAR_FIELD_ROLE,
                kind=kind,
                source_spec={"source": source_spec, "field": field},
                paths=tuple(streams[field]),
            )
        )
    return specs


@contextmanager
def _local_sidecars(
    paths: Sequence[str], indexes_root: Optional[Pathlike]
) -> Generator[Dict[str, Path], None, None]:
    """Download remote sidecars to a temporary directory while building a pack."""
    with tempfile.TemporaryDirectory(prefix="lhotse-shar-pack-") as directory:
        index_paths = {}
        for path in paths:
            index_path = index_file_path(path, indexes_root)
            if is_valid_url(index_path):
                local_path = Path(directory) / f"{len(index_paths)}.idx"
                downloads = [(index_path, local_path)]
                if _is_gzip_jsonl(path):
                    downloads.append(
                        (
                            gzip_index_file_path(path, index_path=index_path),
                            gzip_index_file_path(path, index_path=local_path),
                        )
                    )
                for source, destination in downloads:
                    with open_best(source, "rb") as src, open(destination, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                index_path = local_path
            index_path = Path(index_path)
            if not index_path.is_file():
                raise FileNotFoundError(
                    f"Missing .idx sidecar for '{path}': {index_path}. "
                    "Create it with `lhotse index shar`."
                )
            index_paths[path] = index_path
        yield index_paths


def _check_record_counts(
    streams: Dict[str, List[str]],
    index_paths: Dict[str, Path],
    cache: Dict[str, int],
) -> None:
    """Every field shard must hold one record per cut of the corresponding cuts shard."""

    def count(path: str) -> int:
        if path not in cache:
            # Sidecars are uint64 offsets followed by a size sentinel.
            cache[path] = index_paths[path].stat().st_size // _U64.size - 1
        return cache[path]

    for field, shards in streams.items():
        if field == "cuts":
            continue
        for shard_idx, (cuts_path, path) in enumerate(zip(streams["cuts"], shards)):
            if count(path) != count(cuts_path):
                raise ValueError(
                    f"Shar field '{field}' shard {shard_idx} ('{path}') has {count(path)} records but "
                    f"'{cuts_path}' has {count(cuts_path)} cuts."
                )


def _source_sizes(paths: Sequence[str], index_paths: Dict[str, Path]) -> Dict[str, int]:
    """Check remote JSONL sizes and return tar sizes after verifying zero padding."""
    sizes = {}
    for path in paths:
        is_tar = indexed_path_kind(path) == "tar"
        if not is_tar and (not is_valid_url(path) or _is_gzip_jsonl(path)):
            continue
        sidecar = index_paths[path]
        with open(sidecar, "rb") as f:
            f.seek(-_U64.size, os.SEEK_END)
            (sentinel,) = _U64.unpack(f.read(_U64.size))
        if is_valid_url(path):
            with _open_for_indexed_read(path) as source:
                size = source.seek(0, os.SEEK_END)
        else:
            size = Path(path).stat().st_size
        if sentinel > size:
            raise ValueError(
                f"Sidecar {sidecar} ends at byte {sentinel}, beyond the end of '{path}' ({size} bytes)."
            )
        if sentinel < size:
            if not is_tar:
                raise ValueError(
                    f"Sidecar {sidecar} ends at byte {sentinel}, but '{path}' has {size} bytes."
                )
            _check_zero_tail(path, sentinel, size)
            sizes[path] = size
    return sizes


def _check_zero_tail(path: str, start: int, end: int) -> None:
    with _open_for_indexed_read(path) as f:
        f.seek(start)
        remaining = end - start
        while remaining > 0:
            chunk = f.read(min(remaining, 1 << 20))
            if not chunk:
                raise ValueError(f"'{path}' changed while it was being verified.")
            if chunk.count(0) != len(chunk):
                raise ValueError(
                    f"'{path}' has non-zero bytes after the last indexed sample (offset {start}); "
                    "its sidecar is stale or the archive was appended to. Rebuild the sidecar and retry."
                )
            remaining -= len(chunk)
