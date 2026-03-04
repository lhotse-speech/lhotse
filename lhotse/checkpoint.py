"""
Iterator graph traversal and checkpoint utilities for resumable dataloading.

This module provides:

* :func:`collect_state_dict` — recursively collect state from all
  :class:`~lhotse.lazy.StatefulIterator` nodes in a lazy iterator graph.
* :func:`restore_state_dict` — recursively restore state to all nodes.
* :class:`DataloaderCheckpoint` — a serializable container for the full
  dataloader state (per-worker iterator states + sampler state).
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from lhotse.lazy import StatefulIterator
from lhotse.utils import Pathlike

__all__ = [
    "collect_state_dict",
    "restore_state_dict",
    "DataloaderCheckpoint",
    "register_origin_loader",
    "reload_from_origin",
]

# Attribute names for child references (unified in Phase 2)
_SINGLE_CHILD = "source"
_MULTI_CHILDREN = "sources"


# ---------------------------------------------------------------------------
# Origin registry — extensible loaders for checkpoint restore
# ---------------------------------------------------------------------------

_ORIGIN_LOADERS: Dict[str, Callable[[str, int], Any]] = {}


def register_origin_loader(
    origin_type: str, loader_fn: Callable[[str, int], Any]
) -> None:
    """
    Register a loader for a custom origin type.

    *loader_fn* signature: ``(path: str, idx: int) -> Any``.
    Each call should be self-contained (open, read one item, close).
    """
    _ORIGIN_LOADERS[origin_type] = loader_fn


def reload_from_origin(origin) -> Any:
    """Re-read a single item from its origin coordinates."""
    type_, path, idx = origin
    if type_ not in _ORIGIN_LOADERS:
        raise ValueError(
            f"Unknown origin type '{type_}'. Register a loader with "
            f"register_origin_loader('{type_}', fn)."
        )
    return _ORIGIN_LOADERS[type_](path, idx)


def _load_lhotse_origin(path: str, idx: int):
    from lhotse.indexing import IndexedJsonlReader
    from lhotse.serialization import deserialize_item

    reader = IndexedJsonlReader(path)
    return deserialize_item(reader[idx])


def _load_lhotse_shar_origin(path: str, idx: int):
    from lhotse.shar.readers.indexed import LazyIndexedSharIterator

    reader = LazyIndexedSharIterator(in_dir=path)
    return reader[idx]


def _load_lhotse_shar_fields_origin(path_json: str, idx: int):
    import json

    from lhotse.indexing import IndexedJsonlReader, IndexedTarReader
    from lhotse.serialization import deserialize_item, extension_contains

    shard_paths = json.loads(path_json)

    cut = deserialize_item(IndexedJsonlReader(shard_paths["cuts"])[idx])
    for field, field_path in shard_paths.items():
        if field == "cuts":
            continue
        if extension_contains(".tar", field_path):
            maybe_manifest, data_path = IndexedTarReader(field_path)[idx]
            if maybe_manifest is not None:
                setattr(cut, field, maybe_manifest)
        else:
            item = IndexedJsonlReader(field_path)[idx]
            if field in item:
                setattr(cut, field, item[field])
    return cut


register_origin_loader("lhotse", _load_lhotse_origin)
register_origin_loader("lhotse_shar", _load_lhotse_shar_origin)
register_origin_loader("lhotse_shar_fields", _load_lhotse_shar_fields_origin)


def _rng_state_to_json(rng_state) -> list:
    """Convert a ``random.Random.getstate()`` tuple to JSON-safe lists."""
    version, internalstate, gauss_next = rng_state
    return [version, list(internalstate), gauss_next]


def _rng_state_from_json(data) -> tuple:
    """Reconstruct a ``random.Random`` state tuple from JSON data."""
    version, internalstate, gauss_next = data
    return (version, tuple(internalstate), gauss_next)


# ---------------------------------------------------------------------------
# Graph traversal
# ---------------------------------------------------------------------------


def collect_state_dict(root) -> dict:
    """
    Recursively collect state from all :class:`StatefulIterator` nodes
    in the lazy iterator graph rooted at *root*.

    Returns a nested dict with keys:

    * ``"_type"`` — the class name of the node
    * ``"_state"`` — the node's own ``state_dict()`` (if it is a
      :class:`StatefulIterator`)
    * ``"source"`` — child state dict (if the node has a ``source`` attribute)
    * ``"sources"`` — list of child state dicts (if the node has a ``sources``
      attribute)
    """
    result: Dict[str, Any] = {"_type": type(root).__name__}
    has_children = hasattr(root, _SINGLE_CHILD) or hasattr(root, _MULTI_CHILDREN)

    if isinstance(root, StatefulIterator):
        result["_state"] = root.state_dict()
    elif has_children:
        # Any node that participates in the iterator graph (has source/sources)
        # MUST be a StatefulIterator so that checkpoint state is complete.
        # If it's not, the checkpoint would silently skip this node's state,
        # producing incorrect restoration.  This catches both known
        # non-checkpointable classes (LazyShuffler, etc.) and any future
        # classes that forget to inherit StatefulIterator.
        raise NotImplementedError(
            f"{type(root).__name__} does not support checkpointing "
            f"(it has child iterators but is not a StatefulIterator). "
            f"Either make it a StatefulIterator or remove it from the "
            f"pipeline before checkpointing."
        )

    # Recurse into children
    if hasattr(root, _SINGLE_CHILD):
        child = getattr(root, _SINGLE_CHILD)
        result[_SINGLE_CHILD] = collect_state_dict(child)

    if hasattr(root, _MULTI_CHILDREN):
        children = getattr(root, _MULTI_CHILDREN)
        result[_MULTI_CHILDREN] = [collect_state_dict(c) for c in children]

    return result


def restore_state_dict(root, state: dict) -> None:
    """
    Recursively restore state to all :class:`StatefulIterator` nodes
    in the lazy iterator graph rooted at *root*.

    When *root* is a :class:`StatefulIterator`, its ``load_state_dict``
    already restores any :class:`StatefulIterator` children, so graph
    traversal only recurses into children that are **not** themselves
    :class:`StatefulIterator` (e.g. :class:`LazyShuffler`).  This avoids
    double-restoration which would leave stale ``_restored`` flags on
    inner nodes.

    Validates that the type name at each node matches the saved state.

    :raises TypeError: if a node's type name does not match the saved state.
    """
    expected_type = state.get("_type")
    actual_type = type(root).__name__
    if expected_type is not None and actual_type != expected_type:
        raise TypeError(
            f"Type mismatch during state restoration: "
            f"expected '{expected_type}', got '{actual_type}'."
        )

    root_is_stateful = isinstance(root, StatefulIterator)

    if root_is_stateful and "_state" in state:
        root.load_state_dict(state["_state"])

    # Recurse into children — but skip children that were already restored
    # by the parent's load_state_dict (i.e. StatefulIterator children of a
    # StatefulIterator parent).
    if _SINGLE_CHILD in state and hasattr(root, _SINGLE_CHILD):
        child = getattr(root, _SINGLE_CHILD)
        if not (root_is_stateful and isinstance(child, StatefulIterator)):
            restore_state_dict(child, state[_SINGLE_CHILD])

    if _MULTI_CHILDREN in state and hasattr(root, _MULTI_CHILDREN):
        children = getattr(root, _MULTI_CHILDREN)
        child_states = state[_MULTI_CHILDREN]
        if len(children) != len(child_states):
            raise ValueError(
                f"Number of children mismatch during state restoration: "
                f"expected {len(child_states)}, got {len(children)}."
            )
        if not root_is_stateful:
            for child, child_state in zip(children, child_states):
                restore_state_dict(child, child_state)


# ---------------------------------------------------------------------------
# DataloaderCheckpoint
# ---------------------------------------------------------------------------


@dataclass
class DataloaderCheckpoint:
    """
    Serializable container for the full dataloader checkpoint state.

    Contains per-worker iterator graph states and the sampler state,
    along with metadata needed to validate compatibility on restore.
    """

    num_workers: int
    world_size: int
    rank: int
    worker_states: List[dict] = field(default_factory=list)
    sampler_state: dict = field(default_factory=dict)

    def save(self, path: Pathlike) -> None:
        """Serialize the checkpoint to a JSON file."""
        path = Path(path)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=_json_serializer)

    @classmethod
    def load(cls, path: Pathlike) -> "DataloaderCheckpoint":
        """Deserialize a checkpoint from a JSON file."""
        path = Path(path)
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    def validate(self, num_workers: int, world_size: int, rank: int = 0) -> None:
        """
        Validate that the checkpoint is compatible with the current
        dataloader configuration.

        :raises ValueError: if ``num_workers``, ``world_size``, or ``rank``
            differ from the checkpoint.
        """
        if self.num_workers != num_workers:
            raise ValueError(
                f"Checkpoint num_workers={self.num_workers} does not match "
                f"current num_workers={num_workers}."
            )
        if self.world_size != world_size:
            raise ValueError(
                f"Checkpoint world_size={self.world_size} does not match "
                f"current world_size={world_size}."
            )
        if self.rank != rank:
            raise ValueError(
                f"Checkpoint rank={self.rank} does not match " f"current rank={rank}."
            )


def _json_serializer(obj):
    """Fallback serializer for JSON — handles tuples (from RNG state) etc."""
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
