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

* a Shar directory, identified by ``str(Path(in_dir))`` (the path is not resolved, so use the same spelling when
  writing and reading; a trailing slash does not matter);
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
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from lhotse.index_pack import (
    IndexPack,
    IndexPackCollectionSpec,
    index_pack_collection_key,
    open_index_pack,
    write_index_pack,
)
from lhotse.indexing import (
    index_file_path,
    indexed_path_kind,
    read_tar_pair_at,
)
from lhotse.packed_lazy import read_packed_range
from lhotse.serialization import decode_json_line
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

    The sidecars must exist (create them with ``lhotse index shar`` or ``SharWriter(create_index=True)``).
    The following is verified while packing, and a violation raises an error:

    * every shard is a local, randomly addressable file: ``.jsonl`` / ``.jsonl.gz`` cuts, ``.tar`` fields
      (or ``.jsonl`` / ``.jsonl.gz`` fields); compressed tars cannot be addressed and are rejected;
    * every field has as many shards as the cuts, and every shard has as many records as the corresponding cuts shard;
    * no source file is newer than its sidecar;
    * when a tar sidecar ends before the end of the file, the remaining bytes are the zero padding of the tar
      end-of-archive marker (they are verified to be all zeros; the pack then records the file size).

    :param output_path: where to write the pack.
    :param sources: one Shar source, or a sequence of them (see above).
    :param indexes_root: optional root that mirrors the data layout for sidecar lookup, as in
        :func:`lhotse.indexing.index_file_path`. Sidecars must be local files.
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

    record_counts = {}
    for _, streams in resolved:
        _check_record_counts(streams, indexes_root, record_counts)
    source_size_overrides = _tar_padding_source_sizes(
        sorted({p for _, streams in resolved for p in _tar_shards(streams)}),
        indexes_root,
    )

    return write_index_pack(
        output_path,
        specs,
        indexes_root=indexes_root,
        overwrite=overwrite,
        source_size_overrides=source_size_overrides,
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
        return {"in_dir": str(Path(in_dir))}
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
        from lhotse.shar.readers.discovery import _discover_fields

        source_spec = shar_source_spec(in_dir=source)
        try:
            _, found = _discover_fields(Path(source))
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
    if path.startswith("pipe:") or is_valid_url(path):
        raise ValueError(
            f"Index packs require local shard files, got '{path}' (field '{field}')."
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


def _tar_shards(streams: Dict[str, List[str]]) -> List[str]:
    return [p for shards in streams.values() for p in shards if _is_tar(p)]


def _is_tar(path: str) -> bool:
    return indexed_path_kind(path) == "tar"


def _sidecar_path(path: str, indexes_root: Optional[Pathlike]) -> Path:
    idx = index_file_path(path, indexes_root)
    if is_valid_url(str(idx)):
        raise ValueError(
            f"Index packs are built from local sidecars, got '{idx}' for '{path}'."
        )
    if not Path(idx).exists():
        raise FileNotFoundError(
            f"Missing .idx sidecar for '{path}': {idx}. Create it with `lhotse index shar`."
        )
    return Path(idx)


def _check_record_counts(
    streams: Dict[str, List[str]],
    indexes_root: Optional[Pathlike],
    cache: Dict[str, int],
) -> None:
    """Every field shard must hold one record per cut of the corresponding cuts shard."""

    def count(path: str) -> int:
        if path not in cache:
            # Sidecars are uint64 offsets followed by a size sentinel.
            cache[path] = (
                _sidecar_path(path, indexes_root).stat().st_size // _U64.size - 1
            )
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


def _tar_padding_source_sizes(
    tar_paths: Sequence[str], indexes_root: Optional[Pathlike]
) -> Dict[str, int]:
    """
    The sentinel of a tar sidecar may end at the last member pair, while the file additionally carries the zero
    blocks of the tar end-of-archive marker. Verify that the tail is all zeros and return the file sizes to record.
    """
    sizes = {}
    for path in tar_paths:
        sidecar = _sidecar_path(path, indexes_root)
        with open(sidecar, "rb") as f:
            f.seek(-_U64.size, os.SEEK_END)
            (sentinel,) = _U64.unpack(f.read(_U64.size))
        size = Path(path).stat().st_size
        if sentinel > size:
            raise ValueError(
                f"Sidecar {sidecar} ends at byte {sentinel}, beyond the end of '{path}' ({size} bytes)."
            )
        if sentinel < size:
            _check_zero_tail(path, sentinel, size)
            sizes[path] = size
    return sizes


def _check_zero_tail(path: str, start: int, end: int) -> None:
    with open(path, "rb") as f:
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
