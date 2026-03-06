import bisect
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from lhotse.cut import Cut
from lhotse.lazy import (
    IteratorNode,
    LazyIteratorChain,
    attach_graph_origin,
    attach_origin,
    is_dill_enabled,
    normalize_graph_token,
)
from lhotse.serialization import deserialize_item, extension_contains
from lhotse.shar.readers.lazy import _discover_fields, _is_local_uncompressed
from lhotse.utils import Pathlike, exactly_one_not_null


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

    Requires that all cuts JSONL shards are **uncompressed local files**.
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
        seed: int = 42,
        split_for_dataloading: bool = False,
        index_path: Optional[Union[Pathlike, Dict[str, Sequence[Pathlike]]]] = None,
    ) -> None:
        assert exactly_one_not_null(
            fields, in_dir
        ), "To read Lhotse Shar format, provide either 'in_dir' or 'fields' argument."

        if in_dir is not None:
            self.in_dir = Path(in_dir)
            self.fields, self.streams = _discover_fields(self.in_dir)
        else:
            assert (
                "cuts" in fields
            ), "To initialize Shar reader, please provide the value for key 'cuts' in 'fields'."
            self.fields = set(fields.keys())
            self.fields.remove("cuts")
            self.streams = fields

        self.num_shards = len(self.streams["cuts"])
        for field in self.fields:
            assert (
                len(self.streams[field]) == self.num_shards
            ), f"Expected {self.num_shards} shards available for field '{field}' but found {len(self.streams[field])}: {self.streams[field]}"

        self.shards = [
            {field: self.streams[field][shard_idx] for field in self.streams}
            for shard_idx in range(self.num_shards)
        ]

        # Validate that all cuts JSONL shards are uncompressed local files.
        for cuts_path in self.streams["cuts"]:
            if not _is_local_uncompressed(cuts_path):
                raise ValueError(
                    f"LazyIndexedSharIterator requires uncompressed local cuts "
                    f"JSONL shards, but got: {cuts_path}"
                )

        # ----- Resolve index_path into per-shard per-field index paths -----
        self._index_streams: Optional[Dict[str, List[Optional[Path]]]] = None
        self._raw_index_path = index_path  # kept for pickling

        if index_path is not None:
            if in_dir is not None:
                # index_path is a directory
                idx_dir = Path(index_path)
                self._index_streams = {}
                for field_name in self.streams:
                    per_shard = []
                    for data_p in self.streams[field_name]:
                        ip = idx_dir / (Path(data_p).name + ".idx")
                        per_shard.append(ip)
                    self._index_streams[field_name] = per_shard
            else:
                # index_path is a dict
                if not isinstance(index_path, dict):
                    raise TypeError(
                        "When using 'fields' mode, 'index_path' must be a dict "
                        f"mapping field names to lists of .idx paths, got {type(index_path)}."
                    )
                for key in index_path:
                    if key not in self.streams and key != "cuts":
                        raise ValueError(
                            f"index_path key '{key}' does not match any field. "
                            f"Expected keys from: {set(self.streams.keys())}"
                        )
                    n_idx = len(index_path[key])
                    n_data = len(self.streams.get(key, self.streams.get("cuts", [])))
                    if n_idx != n_data:
                        raise ValueError(
                            f"index_path['{key}'] has {n_idx} entries but "
                            f"there are {n_data} data shards."
                        )
                self._index_streams = {
                    k: [Path(p) for p in v] for k, v in index_path.items()
                }

        self.shuffle = shuffle
        self.seed = seed
        self.split_for_dataloading = split_for_dataloading
        self.epoch = 0

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

        # Iteration state
        self._position = 0
        self._restored = False

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

        # Attach origin for checkpoint reload.
        ip_str = str(self._raw_index_path) if self._raw_index_path is not None else None
        if hasattr(self, "in_dir"):
            attach_origin(
                cut,
                ("lhotse_shar", str(self.in_dir), global_idx, ip_str, item_epoch),
            )
        else:
            # Encode the per-shard path mapping so the loader can
            # reconstruct indexed readers for this specific shard.
            import json

            shard_paths = {k: str(v) for k, v in self.shards[shard_idx].items()}
            if self._index_streams is not None:
                # Include per-field index paths for this shard under "_index".
                idx_paths = {}
                for f_name, f_paths in self._index_streams.items():
                    idx_paths[f_name] = str(f_paths[shard_idx])
                shard_paths["_index"] = idx_paths
            attach_origin(
                cut,
                ("lhotse_shar_fields", json.dumps(shard_paths), pos, item_epoch),
            )

        return cut

    def _get_worker_indices(self) -> List[int]:
        """Return the global indices assigned to this worker/node."""
        indices = list(range(self._total_len))
        if self.split_for_dataloading:
            from lhotse.shar.readers.utils import split_by_node, split_by_worker

            indices = split_by_node(indices)
            indices = split_by_worker(indices)
        return indices

    def __iter__(self):
        from lhotse.indexing import LazyShuffledRange

        indices = self._get_worker_indices()
        n = len(indices)

        if self._restored:
            self._restored = False
            start = self._position
        else:
            start = 0
            self._position = 0

        if self.shuffle:
            shuffled = LazyShuffledRange(n, seed=self.seed + self.epoch)
            for i in range(start, n):
                self._position = i + 1
                yield self[indices[shuffled[i]]]
        else:
            for i in range(start, n):
                self._position = i + 1
                yield self[indices[i]]

        self.epoch += 1

    def state_dict(self) -> dict:
        return {
            "position": self._position,
            "epoch": self.epoch,
            "shuffle": self.shuffle,
            "seed": self.seed,
        }

    def load_state_dict(self, sd: dict) -> None:
        self._position = sd["position"]
        self.epoch = sd["epoch"]
        self._restored = True

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
