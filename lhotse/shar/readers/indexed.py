import bisect
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from lhotse.cut import Cut
from lhotse.dataset.dataloading import PartitionedIndexedIterator, resolve_seed
from lhotse.indexing import (
    create_jsonl_index,
    create_tar_index,
    index_exists,
    validate_indexed_access,
)
from lhotse.lazy import (
    IteratorNode,
    LazyIteratorChain,
    attach_graph_origin,
    is_dill_enabled,
    normalize_graph_token,
)
from lhotse.serialization import deserialize_item, extension_contains
from lhotse.shar.readers.lazy import _discover_fields
from lhotse.utils import Pathlike, exactly_one_not_null, is_valid_url


class LazyIndexedSharIterator(IteratorNode):
    """
    Indexed random-access reader for the Lhotse Shar format.

    Unlike :class:`LazySharIterator` (which streams sequentially through
    shards), this class uses binary ``.idx`` indexes for O(1) random access
    to any cut across all shards.  It is modeled after
    :class:`~lhotse.lazy.LazyIndexedManifestIterator`.

    Supports:

    * ``__getitem__(idx)`` — O(1) random access by global index.
    * ``__len__()`` — total number of cuts across all shards.
    * ``__iter__()`` — sequential or shuffled iteration via ``__getitem__``.
    * ``state_dict()`` / ``load_state_dict()`` — checkpoint/restore.

    Requires uncompressed, seekable JSONL/tar shards for every requested field.
    These may live on the local filesystem or on supported remote/object-store
    backends, as long as the underlying reader can perform indexed reads.
    Binary ``.idx`` indexes are created automatically if missing.

    :param fields: a dict whose keys specify which fields to load,
        and values are lists of shards (either paths or shell commands).
        The field "cuts" pointing to CutSet shards always has to be present.
    :param in_dir: path to a directory created with ``SharWriter`` with
        all the shards in a single place. Can be used instead of ``fields``.
    :param shuffle: if ``True``, iteration uses a Feistel-network
        permutation for true random access across all shards.
    :param seed: seed for the shuffle permutation.
    :param split_for_dataloading: if ``True``, each PyTorch DataLoader
        worker (and DDP node) iterates a unique slice of the global
        index range, avoiding data duplication.
    :param index_path: optional location of ``.idx`` files stored
        separately from the data.  Accepted shapes:

        * A directory path (when ``in_dir`` is used) — for each data file
          ``in_dir/<name>``, the index is expected at
          ``index_path/<name>.idx``.
        * A dict (when ``fields`` is used) — keys must match ``fields``
          keys, and each value is a list of ``.idx`` file paths
          (one per shard, matching the order in ``fields``).
    """

    is_checkpointable = True

    def __init__(
        self,
        fields: Optional[Dict[str, Sequence[Pathlike]]] = None,
        in_dir: Optional[Pathlike] = None,
        *,
        shuffle: bool = False,
        seed: Union[int, str] = 42,
        split_for_dataloading: bool = False,
        index_path: Optional[Union[Pathlike, Dict[str, Sequence[Pathlike]]]] = None,
        indexes_root: Optional[Pathlike] = None,
        lazy: bool = False,
    ) -> None:
        if index_path is not None and indexes_root is not None:
            raise ValueError(
                "Pass either 'index_path' (explicit per-shard paths or directory) "
                "or 'indexes_root' (a root that mirrors data layout), not both."
            )
        self.in_dir = Path(in_dir) if in_dir is not None else None
        self.fields, self.streams = self._resolve_streams(fields=fields, in_dir=in_dir)

        self.num_shards = len(self.streams["cuts"])
        for field in self.fields:
            assert (
                len(self.streams[field]) == self.num_shards
            ), f"Expected {self.num_shards} shards available for field '{field}' but found {len(self.streams[field])}: {self.streams[field]}"

        self.shards = [
            {field: self.streams[field][shard_idx] for field in self.streams}
            for shard_idx in range(self.num_shards)
        ]

        # ----- Resolve index_path into per-shard per-field index paths -----
        if indexes_root is not None:
            index_path = _index_path_from_indexes_root(self.streams, indexes_root)
        self._index_streams: Optional[Dict[str, List[Optional[Pathlike]]]] = None
        self._raw_index_path = index_path  # kept for pickling
        self._index_streams = self._resolve_index_streams(
            streams=self.streams,
            index_path=index_path,
            in_dir=in_dir,
        )
        self._validate_indexed_streams(
            streams=self.streams,
            index_streams=self._index_streams,
            auto_create_index=True,
        )

        self.shuffle = shuffle
        self.seed = seed
        # ``split_for_dataloading`` is preserved for backwards-compat with
        # ``CutSet.from_shar(split_for_dataloading=...)`` callers, but is
        # no longer load-bearing: partitioning is delegated to
        # ``PartitionedIndexedIterator`` which reads ``(rank, world_size,
        # worker_id)`` via ``get_worker_partition()``. The old per-cut
        # stride partition (split_by_node + split_by_worker) is replaced
        # because it didn't track topology in state_dict, causing silent
        # divergence on resume under a different (world_size, num_workers).
        self.split_for_dataloading = split_for_dataloading
        self._lazy = lazy
        self.epoch = 0
        self._iter_state = PartitionedIndexedIterator(
            shuffle=self.shuffle, seed=resolve_seed(self.seed) if isinstance(self.seed, int) else 0
        )

        # Build indexed readers for cuts JSONL shards and compute lengths.
        from lhotse.indexing import IndexedJsonlReader

        cuts_idx_paths = (
            self._index_streams.get("cuts") if self._index_streams else None
        )
        self._cuts_readers: List[IndexedJsonlReader] = [
            IndexedJsonlReader(
                p,
                index_path=cuts_idx_paths[i] if cuts_idx_paths else None,
            )
            for i, p in enumerate(self.streams["cuts"])
        ]
        self._shard_lens = [len(r) for r in self._cuts_readers]

        # Cumulative lengths for global -> (shard, local) mapping.
        self._cum_lens: List[int] = []
        total = 0
        for sl in self._shard_lens:
            self._cum_lens.append(total)
            total += sl
        self._total_len = total

        # Lazily-created indexed readers for non-cuts fields.
        self._indexed_readers: Optional[Dict[int, dict]] = None

    @staticmethod
    def _join_index_dir(index_dir: Pathlike, filename: str) -> Pathlike:
        if isinstance(index_dir, Path):
            return index_dir / filename
        index_dir = str(index_dir)
        if is_valid_url(index_dir):
            return f"{index_dir.rstrip('/')}/{filename}"
        return Path(index_dir) / filename

    @classmethod
    def _resolve_streams(
        cls,
        *,
        fields: Optional[Dict[str, Sequence[Pathlike]]],
        in_dir: Optional[Pathlike],
    ) -> Tuple[set, Dict[str, Sequence[Pathlike]]]:
        assert exactly_one_not_null(
            fields, in_dir
        ), "To read Lhotse Shar format, provide either 'in_dir' or 'fields' argument."
        if in_dir is not None:
            _, streams = _discover_fields(Path(in_dir))
            field_names = set(streams.keys())
            field_names.remove("cuts")
            return field_names, streams
        assert (
            "cuts" in fields
        ), "To initialize Shar reader, please provide the value for key 'cuts' in 'fields'."
        field_names = set(fields.keys())
        field_names.remove("cuts")
        return field_names, fields

    @classmethod
    def _resolve_index_streams(
        cls,
        *,
        streams: Dict[str, Sequence[Pathlike]],
        index_path: Optional[Union[Pathlike, Dict[str, Sequence[Pathlike]]]],
        in_dir: Optional[Pathlike],
    ) -> Optional[Dict[str, List[Optional[Pathlike]]]]:
        if index_path is None:
            return None
        if in_dir is not None:
            index_streams = {}
            for field_name, shard_paths in streams.items():
                index_streams[field_name] = [
                    cls._join_index_dir(index_path, Path(str(data_p)).name + ".idx")
                    for data_p in shard_paths
                ]
            return index_streams
        if not isinstance(index_path, dict):
            raise TypeError(
                "When using 'fields' mode, 'index_path' must be a dict "
                f"mapping field names to lists of .idx paths, got {type(index_path)}."
            )
        for key, idx_paths in index_path.items():
            if key not in streams:
                raise ValueError(
                    f"index_path key '{key}' does not match any field. "
                    f"Expected keys from: {set(streams.keys())}"
                )
            if len(idx_paths) != len(streams[key]):
                raise ValueError(
                    f"index_path['{key}'] has {len(idx_paths)} entries but "
                    f"there are {len(streams[key])} data shards."
                )
        return {k: list(v) for k, v in index_path.items()}

    @classmethod
    def _validate_indexed_streams(
        cls,
        *,
        streams: Dict[str, Sequence[Pathlike]],
        index_streams: Optional[Dict[str, List[Optional[Pathlike]]]],
        auto_create_index: bool,
    ) -> None:
        for field_name, shard_paths in streams.items():
            expected_kind = "jsonl" if field_name == "cuts" else None
            for shard_idx, path in enumerate(shard_paths):
                context = (
                    f"LazyIndexedSharIterator field '{field_name}' shard {shard_idx}"
                )
                kind = validate_indexed_access(
                    path, kind=expected_kind, context=context
                )
                idx_path = None
                if index_streams is not None and field_name in index_streams:
                    idx_path = index_streams[field_name][shard_idx]
                if index_exists(path, index_path=idx_path):
                    continue
                if not auto_create_index:
                    raise FileNotFoundError(
                        f"{context} is missing an index file. "
                        f"Expected it at {idx_path if idx_path is not None else str(path) + '.idx'}."
                    )
                if kind == "jsonl":
                    create_jsonl_index(path, output_path=idx_path)
                else:
                    create_tar_index(path, output_path=idx_path)

    @classmethod
    def supports_configuration(
        cls,
        *,
        fields: Optional[Dict[str, Sequence[Pathlike]]] = None,
        in_dir: Optional[Pathlike] = None,
        index_path: Optional[Union[Pathlike, Dict[str, Sequence[Pathlike]]]] = None,
        indexes_root: Optional[Pathlike] = None,
    ) -> bool:
        if index_path is not None and indexes_root is not None:
            return False
        try:
            _, streams = cls._resolve_streams(fields=fields, in_dir=in_dir)
            if indexes_root is not None:
                index_path = _index_path_from_indexes_root(streams, indexes_root)
            index_streams = cls._resolve_index_streams(
                streams=streams,
                index_path=index_path,
                in_dir=in_dir,
            )
            cls._validate_indexed_streams(
                streams=streams,
                index_streams=index_streams,
                auto_create_index=False,
            )
            return True
        except (AssertionError, TypeError, ValueError, FileNotFoundError):
            return False

    @property
    def is_indexed(self) -> bool:
        return True

    @property
    def has_constant_time_access(self) -> bool:
        return True

    def __len__(self) -> int:
        return self._total_len

    def _resolve_index(self, idx: int) -> Tuple[int, int]:
        """Map a global index to ``(shard_idx, position_within_shard)``."""
        if idx < 0:
            idx += self._total_len
        if idx < 0 or idx >= self._total_len:
            raise IndexError(
                f"index {idx} out of range for LazyIndexedSharIterator "
                f"with {self._total_len} cuts"
            )
        # Binary search over cumulative lengths.
        shard_idx = bisect.bisect_right(self._cum_lens, idx) - 1
        return shard_idx, idx - self._cum_lens[shard_idx]

    def _ensure_indexed_readers(self, shard_idx: int) -> dict:
        """Lazily create and cache indexed readers for *shard_idx*."""
        from lhotse.indexing import IndexedJsonlReader, IndexedTarReader

        if self._indexed_readers is None:
            self._indexed_readers = {}
        if shard_idx in self._indexed_readers:
            return self._indexed_readers[shard_idx]

        shard = self.shards[shard_idx]
        readers = {}
        for field in self.fields:
            path = shard[field]
            ip = (
                self._index_streams[field][shard_idx]
                if self._index_streams and field in self._index_streams
                else None
            )
            if extension_contains(".tar", path):
                readers[field] = IndexedTarReader(path, index_path=ip)
            else:
                readers[field] = IndexedJsonlReader(path, index_path=ip)

        self._indexed_readers[shard_idx] = readers
        return readers

    def __getitem__(self, idx: Any) -> Cut:
        """O(1) random access to a cut by global index."""
        idx = normalize_graph_token(idx)
        item_epoch = self.epoch
        if isinstance(idx, tuple):
            if len(idx) != 2:
                raise TypeError(
                    "LazyIndexedSharIterator expects graph restore tokens shaped "
                    "like (global_index, shar_epoch)."
                )
            idx, item_epoch = idx

        shard_idx, pos = self._resolve_index(idx)

        # Read and deserialize the cut.
        cut = deserialize_item(self._cuts_readers[shard_idx][pos])

        # Attach field data from tar / JSONL readers.
        if self.fields:
            from lhotse.indexing import IndexedTarReader

            readers = self._ensure_indexed_readers(shard_idx)
            for field in self.fields:
                reader = readers[field]
                if isinstance(reader, IndexedTarReader):
                    if self._lazy:
                        # Lazy mode: emit a Shar pointer derived purely from
                        # the .idx offset array — zero tar reads at iter time.
                        offset, end_offset = reader.member_byte_range(pos)
                        from lhotse.shar.utils import fill_shar_placeholder_lazy

                        fill_shar_placeholder_lazy(
                            cut,
                            field=field,
                            tar_path=str(reader.path),
                            offset=offset,
                            end_offset=end_offset,
                        )
                    else:
                        maybe_manifest, data_path = reader[pos]
                        if maybe_manifest is not None:
                            assert str(data_path.parent / data_path.stem) == cut.id, (
                                f"Mismatched IDs: cut ID is '{cut.id}' but found "
                                f"data with name '{data_path}' for field {field}"
                            )
                            setattr(cut, field, maybe_manifest)
                else:
                    item = reader[pos]
                    if field in item:
                        setattr(cut, field, item[field])

        cut.shard_origin = self.shards[shard_idx]["cuts"]
        cut.shar_epoch = item_epoch
        global_idx = idx if idx >= 0 else idx + self._total_len
        attach_graph_origin(cut, (global_idx, item_epoch))
        return cut

    def __iter__(self):
        for global_idx in self._iter_state.iterate(self._total_len):
            yield self[global_idx]
        self.epoch += 1

    def state_dict(self) -> dict:
        return {
            **self._iter_state.state_dict(),
            "epoch": self.epoch,
            "shuffle": self.shuffle,
            "seed": self.seed,
            "lazy": self._lazy,
        }

    def load_state_dict(self, sd: dict) -> None:
        self._iter_state.load_state_dict(sd)
        self.epoch = sd.get("epoch", 0)
        # Backward-compat: older state dicts may not carry "lazy".
        if "lazy" in sd:
            self._lazy = bool(sd["lazy"])

    # ------------------------------------------------------------------
    # Pickling — exclude non-picklable caches
    # ------------------------------------------------------------------

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop("_indexed_readers", None)
        d.pop("_cuts_readers", None)
        if is_dill_enabled():
            import dill

            return dill.dumps(d)
        return d

    def __setstate__(self, state):
        if is_dill_enabled():
            import dill

            state = dill.loads(state)
        self.__dict__ = state
        self._indexed_readers = None
        # Re-open cuts readers with index paths if available.
        from lhotse.indexing import IndexedJsonlReader

        cuts_idx_paths = (
            self._index_streams.get("cuts") if self._index_streams else None
        )
        self._cuts_readers = [
            IndexedJsonlReader(
                p,
                index_path=cuts_idx_paths[i] if cuts_idx_paths else None,
            )
            for i, p in enumerate(self.streams["cuts"])
        ]

    def __add__(self, other) -> "LazyIteratorChain":
        return LazyIteratorChain(self, other)


def _index_path_from_indexes_root(
    streams: Dict[str, Sequence[Pathlike]], indexes_root: Pathlike
) -> Dict[str, List[Pathlike]]:
    """Build the per-field, per-shard ``index_path`` dict that
    :meth:`LazyIndexedSharIterator._resolve_index_streams` expects, by
    mirroring each shard's data path under ``indexes_root``.

    Equivalent to ``{field: [index_file_path(p, indexes_root) for p in shards]
    for field, shards in streams.items()}`` — produces e.g.
    ``s3://AMI/lhotse_shar/cuts.000000.jsonl`` →
    ``/tmp/idx/AMI/lhotse_shar/cuts.000000.jsonl.idx``.
    """
    from lhotse.indexing import index_file_path

    return {
        field: [index_file_path(p, indexes_root) for p in shard_paths]
        for field, shard_paths in streams.items()
    }
