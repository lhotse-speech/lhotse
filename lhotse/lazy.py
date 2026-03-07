import os
import random
import types
import warnings
from collections import deque
from contextlib import contextmanager
from functools import partial
from typing import Any, Callable, Iterable, List, Literal, Optional, TypeVar, Union

from lhotse.serialization import (
    LazyMixin,
    decode_json_line,
    deserialize_item,
    open_best,
)
from lhotse.utils import Pathlike, fastcopy, is_module_available

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Dill-backed pickling mixin
# ---------------------------------------------------------------------------


class Dillable:
    """
    Mix-in that will leverage ``dill`` instead of ``pickle``
    when pickling an object.

    It is useful when the user can't avoid ``pickle`` (e.g. in multiprocessing),
    but needs to use unpicklable objects such as lambdas.

    If ``dill`` is not installed, it defers to what ``pickle`` does by default.
    """

    _ENABLED_VALUES = {"1", "True", "true", "yes"}

    def __getstate__(self):
        if is_dill_enabled():
            import dill

            return dill.dumps(self.__dict__)
        else:
            return self.__dict__

    def __setstate__(self, state):
        if is_dill_enabled():
            import dill

            self.__dict__ = dill.loads(state)
        else:
            self.__dict__ = state


# ---------------------------------------------------------------------------
# Iterator node protocol
# ---------------------------------------------------------------------------


class IteratorNode(Dillable, Iterable):
    """
    Base protocol for nodes in Lhotse's lazy iterator graph.

    Conventions for child references:

    * ``self.source``  — single child iterator
    * ``self.sources`` — list of child iterators

    Iterator nodes are not necessarily checkpointable. Nodes that support
    checkpointing should set ``is_checkpointable = True`` and implement
    :meth:`state_dict` and :meth:`load_state_dict`.

    .. warning::
        Instances are **not thread-safe**. Mutable position/restoration flags
        are updated without synchronization. For multi-worker data loading use
        process-based parallelism (the default in PyTorch's ``DataLoader``),
        which gives each worker its own copy.
    """

    is_checkpointable = False
    is_indexed = False
    has_constant_time_access = False

    def state_dict(self) -> dict:
        raise NotImplementedError(
            f"{type(self).__name__} is not checkpointable and does not implement state_dict()."
        )

    def load_state_dict(self, sd: dict) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} is not checkpointable and does not implement load_state_dict()."
        )

    def iter_children(self):
        """Yield child iterators following ``source``/``sources`` conventions."""
        if hasattr(self, "source"):
            yield getattr(self, "source")
        if hasattr(self, "sources"):
            yield from getattr(self, "sources")


def resolve_iterator_source(obj: Iterable) -> Iterable:
    """
    Return the effective iterator payload for graph nodes.

    Manifest wrappers such as ``CutSet`` expose their underlying iterator via
    ``.data``; using it avoids introducing wrapper objects into lazy iterator
    graphs.
    """
    try:
        from lhotse.cut import CutSet
    except Exception:
        return obj
    return obj.data if isinstance(obj, CutSet) else obj


def _try_collect_child_state(obj: Any) -> Optional[dict]:
    if isinstance(obj, IteratorNode):
        if type(obj).state_dict is IteratorNode.state_dict:
            if any(True for _ in obj.iter_children()):
                raise NotImplementedError(
                    f"{type(obj).__name__} does not support checkpointing. "
                    f"Remove it from the pipeline before checkpointing or implement "
                    f"state_dict/load_state_dict."
                )
            return None
        return obj.state_dict()
    if hasattr(obj, "state_dict") and callable(getattr(obj, "state_dict")):
        try:
            return obj.state_dict()
        except Exception:
            return None
    return None


def _try_restore_child_state(obj: Any, state: Optional[dict]) -> None:
    if state is None:
        return
    if isinstance(obj, IteratorNode):
        if type(obj).load_state_dict is IteratorNode.load_state_dict:
            raise NotImplementedError(
                f"{type(obj).__name__} does not support checkpoint restoration. "
                f"Remove it from the pipeline before checkpointing or implement "
                f"state_dict/load_state_dict."
            )
        obj.load_state_dict(state)
        return
    if hasattr(obj, "load_state_dict") and callable(getattr(obj, "load_state_dict")):
        obj.load_state_dict(state)


def _attach_runtime_metadata(item: Any, name: str, value: Any) -> Any:
    """
    Attach iterator runtime metadata without routing through Cut.custom.

    Cut-like objects use ``CustomFieldMixin.__setattr__`` to redirect unknown
    attributes into the serialized ``custom`` field. Graph restore metadata such
    as ``_graph_origin`` must stay process-local and never appear in manifests,
    so we bypass ``__setattr__`` when possible.
    """
    try:
        object.__setattr__(item, name, value)
    except Exception:
        try:
            setattr(item, name, value)
        except Exception:
            pass
    return item


def normalize_graph_token(token: Any) -> Any:
    """Convert JSON-serialized graph tokens back to tuples recursively."""
    if isinstance(token, list):
        return tuple(normalize_graph_token(part) for part in token)
    if isinstance(token, tuple):
        return tuple(normalize_graph_token(part) for part in token)
    return token


def attach_graph_origin(item: Any, token: Any) -> Any:
    return _attach_runtime_metadata(item, "_graph_origin", token)


def get_graph_origin(item: Any) -> Any:
    return getattr(item, "_graph_origin", None)


def maybe_attach_graph_origin(item: Any, token: Any) -> Any:
    if token is None:
        return item
    return attach_graph_origin(item, token)


def require_graph_origin(item: Any, owner: str, what: str = "items") -> Any:
    token = get_graph_origin(item)
    if token is None:
        raise RuntimeError(
            f"{owner} requires '_graph_origin' on {what} from graph-restorable sources."
        )
    return token


def supports_graph_restore(source: Any, *, require_length: bool = False) -> bool:
    if not getattr(source, "has_constant_time_access", False):
        return False
    if not hasattr(source, "__getitem__"):
        return False
    return not require_length or hasattr(source, "__len__")


def resolve_iteration_seed(
    seed: Optional[Union[int, Literal["trng", "randomized"]]]
) -> int:
    from lhotse.dataset.dataloading import resolve_seed

    if seed is None:
        return random.getrandbits(31)
    return resolve_seed(seed)


class AlgorithmMixin(LazyMixin, Iterable):
    """
    Helper base class with methods that are supposed to work identically
    on Lhotse manifest classes such as CutSet, RecordingSet, etc.
    """

    def filter(self, predicate: Callable[[T], bool]):
        """
        Return a new manifest containing only the items that satisfy ``predicate``.
        If the manifest is lazy, the filtering will also be applied lazily.

        :param predicate: a function that takes a cut as an argument and returns bool.
        :return: a filtered manifest.
        """
        cls = type(self)

        if self.is_lazy:
            return cls(LazyFilter(resolve_iterator_source(self), predicate=predicate))

        return cls.from_items(cut for cut in self if predicate(cut))

    def map(self, transform_fn: Callable[[T], T]):
        """
        Apply `transform_fn` to each item in this manifest and return a new manifest.
        If the manifest is opened lazy, the transform is also applied lazily.

        :param transform_fn: A callable (function) that accepts a single item instance
            and returns a new (or the same) instance of the same type.
            E.g. with CutSet, callable accepts ``Cut`` and returns also ``Cut``.
        :return: a new ``CutSet`` with transformed cuts.
        """
        cls = type(self)
        ans = cls(LazyMapper(resolve_iterator_source(self), fn=transform_fn))
        if self.is_lazy:
            return ans
        return ans.to_eager()

    @classmethod
    def mux(
        cls,
        *manifests,
        stop_early: bool = False,
        weights: Optional[List[Union[int, float]]] = None,
        seed: Union[int, Literal["trng", "randomized"]] = 0,
    ):
        """
        Merges multiple manifest iterables into a new iterable by lazily multiplexing them during iteration time.
        If one of the iterables is exhausted before the others, we will keep iterating until all iterables
        are exhausted. This behavior can be changed with ``stop_early`` parameter.

        :param manifests: iterables to be multiplexed.
            They can be either lazy or eager, but the resulting manifest will always be lazy.
        :param stop_early: should we stop the iteration as soon as we exhaust one of the manifests.
        :param weights: an optional weight for each iterable, affects the probability of it being sampled.
            The weights are uniform by default.
            If lengths are known, it makes sense to pass them here for uniform distribution of
            items in the expectation.
        :param seed: the random seed, ensures deterministic order across multiple iterations.
        """
        manifests = [resolve_iterator_source(m) for m in manifests]
        return cls(
            LazyIteratorMultiplexer(
                *manifests, stop_early=stop_early, weights=weights, seed=seed
            )
        )

    @classmethod
    def infinite_mux(
        cls,
        *manifests,
        weights: Optional[List[Union[int, float]]] = None,
        seed: Union[int, Literal["trng", "randomized"]] = 0,
        max_open_streams: Optional[int] = None,
    ):
        """
        Merges multiple manifest iterables into a new iterable by lazily multiplexing them during iteration time.
        Unlike ``mux()``, this method allows to limit the number of max open sub-iterators at any given time.

        To enable this, it performs 2-stage sampling.
        First, it samples with replacement the set of iterators ``I`` to construct a subset ``I_sub``
        of size ``max_open_streams``.
        Then, for each iteration step, it samples an iterator ``i`` from ``I_sub``,
        fetches the next item from it, and yields it.
        Once ``i`` becomes exhausted, it is replaced with a new iterator ``j`` sampled from ``I_sub``.

        .. caution:: Do not use this method with inputs that are infinitely iterable as they will
            silently break the multiplexing property by only using a subset of the input iterables.

        .. caution:: This method is not recommended for multiplexing for a small amount of iterations,
            as it may be much less accurate than ``mux()`` depending on the number of open streams,
            iterable sizes, and the random seed.

        :param manifests: iterables to be multiplexed.
            They can be either lazy or eager, but the resulting manifest will always be lazy.
        :param weights: an optional weight for each iterable, affects the probability of it being sampled.
            The weights are uniform by default.
            If lengths are known, it makes sense to pass them here for uniform distribution of
            items in the expectation.
        :param seed: the random seed, ensures deterministic order across multiple iterations.
        :param max_open_streams: the number of iterables that can be open simultaneously at any given time.
        """
        manifests = [resolve_iterator_source(m) for m in manifests]
        return cls(
            LazyInfiniteApproximateMultiplexer(
                *manifests,
                weights=weights,
                seed=seed,
                max_open_streams=max_open_streams,
            )
        )

    def shuffle(
        self,
        rng: Optional[random.Random] = None,
        buffer_size: int = 10000,
    ):
        """
        Shuffles the elements and returns a shuffled variant of self.
        If the manifest is opened lazily, performs shuffling on-the-fly with a fixed buffer size.

        :param rng: an optional instance of ``random.Random`` for precise control of randomness.
        :return: a shuffled copy of self, or a manifest that is shuffled lazily.
        """
        cls = type(self)

        if rng is None:
            rng = random

        if self.is_lazy:
            return cls(
                LazyShuffler(
                    resolve_iterator_source(self), buffer_size=buffer_size, rng=rng
                )
            )
        else:
            new: List = self.data.copy()
            rng.shuffle(new)
            return cls(new)

    def repeat(self, times: Optional[int] = None, preserve_id: bool = False):
        """
        Return a new, lazily evaluated manifest that iterates over the original elements ``times``
        number of times.

        :param times: how many times to repeat (infinite by default).
        :param preserve_id: when ``True``, we won't update the element ID with repeat number.
        :return: a repeated manifest.
        """
        cls = type(self)
        return cls(
            LazyRepeater(
                resolve_iterator_source(self), times=times, preserve_id=preserve_id
            )
        )

    def __add__(self, other):
        cls = type(self)
        return cls(
            LazyIteratorChain(
                resolve_iterator_source(self), resolve_iterator_source(other)
            )
        )


def is_dill_enabled(_ENABLED_VALUES=frozenset(("1", "True", "true", "yes"))) -> bool:
    """Returns bool indicating if dill-based pickling in Lhotse is enabled or not."""
    return (
        is_module_available("dill")
        and os.environ.get("LHOTSE_DILL_ENABLED", "0") in _ENABLED_VALUES
    )


def set_dill_enabled(value: bool) -> None:
    """Enable or disable dill-based pickling in Lhotse."""
    assert is_module_available("dill"), (
        "Cannot enable dill because dill is not installed. "
        "Please run 'pip install dill' and try again."
    )
    # We use os.environ here so that sub-processes / forks will inherit this value
    os.environ["LHOTSE_DILL_ENABLED"] = "1" if value else "0"


@contextmanager
def dill_enabled(value: bool):
    """
    Context manager that overrides the setting of Lhotse's dill-backed pickling
    and restores the previous value after exit.

    Example::

        >>> import pickle
        ... with dill_enabled(True):
        ...    pickle.dump(CutSet(...).filter(lambda c: c.duration < 5), open("cutset.pickle", "wb"))
    """
    previous = is_dill_enabled()
    set_dill_enabled(value)
    yield
    set_dill_enabled(previous)


class LazyTxtIterator(IteratorNode):
    """
    LazyTxtIterator is a thin wrapper over builtin ``open`` function to
    iterate over lines in a (possibly compressed) text file.
    It can also provide the number of lines via __len__ via fast newlines counting.
    """

    def __init__(self, path: Pathlike, as_text_example: bool = True) -> None:
        self.path = path
        self.as_text_example = as_text_example
        self._len = None

    def __iter__(self):
        from lhotse.cut.text import TextExample

        tot = 0
        with open_best(self.path, "r") as f:
            for line in f:
                line = line.strip()
                if self.as_text_example:
                    line = TextExample(line)
                yield line
                tot += 1
        if self._len is None:
            self._len = tot

    def __len__(self) -> int:
        if self._len is None:
            self._len = count_newlines_fast(self.path)
        return self._len


class LazyJsonlIterator(IteratorNode):
    """
    LazyJsonlIterator provides the ability to read JSON lines as Python dicts.
    It can also provide the number of lines via __len__ via fast newlines counting.
    """

    def __init__(self, path: Pathlike) -> None:
        self.path = path
        self._len = None
        self._position = 0
        self._restored = False

    def __iter__(self):
        start = self._position if self._restored else 0
        self._restored = False
        self._position = start
        tot = 0
        with open_best(self.path, "r") as f:
            for line in f:
                tot += 1
                if tot <= start:
                    continue
                data = decode_json_line(line)
                self._position = tot
                yield data
        if self._len is None:
            self._len = tot

    def __len__(self) -> int:
        if self._len is None:
            self._len = count_newlines_fast(self.path)
        return self._len

    def state_dict(self) -> dict:
        """Return ``{"position": int}``."""
        return {"position": self._position}

    def load_state_dict(self, sd: dict) -> None:
        """Restore position.  Actual seeking happens in ``__iter__``."""
        self._position = sd["position"]
        self._restored = True


class LazyManifestIterator(IteratorNode):
    """
    LazyManifestIterator provides the ability to read Lhotse objects from a
    JSONL file on-the-fly, without reading its full contents into memory.

    This class is designed to be a partial "drop-in" replacement for ordinary dicts
    to support lazy loading of RecordingSet, SupervisionSet and CutSet.
    Since it does not support random access reads, some methods of these classes
    might not work properly.

    Supports checkpointing via :meth:`state_dict` / :meth:`load_state_dict`
    (delegates to the inner :class:`LazyJsonlIterator`).
    """

    is_checkpointable = True

    def __init__(self, path: Pathlike) -> None:
        self.source = LazyJsonlIterator(path)

    @property
    def path(self) -> Pathlike:
        return self.source.path

    def __iter__(self):
        yield from map(deserialize_item, self.source)

    def __len__(self) -> int:
        return len(self.source)

    def __add__(self, other) -> "LazyIteratorChain":
        return LazyIteratorChain(self, other)

    def state_dict(self) -> dict:
        return {"source": self.source.state_dict()}

    def load_state_dict(self, sd: dict) -> None:
        self.source.load_state_dict(sd["source"])


class LazyIndexedManifestIterator(IteratorNode):
    """
    Lazy manifest iterator backed by an :class:`~lhotse.indexing.IndexedJsonlReader`.

    Supports O(1) random access via ``__getitem__`` and optional Feistel-shuffled
    iteration via :class:`~lhotse.indexing.LazyShuffledRange`.

    Unlike :class:`LazyManifestIterator`, this class requires an uncompressed
    JSONL file (the binary ``.idx`` index is created automatically if missing).

    Supports checkpointing via :meth:`state_dict` / :meth:`load_state_dict`.
    """

    is_checkpointable = True

    def __init__(
        self,
        path: Pathlike,
        shuffle: bool = False,
        seed: int = 0,
        index_path: Optional[Pathlike] = None,
    ) -> None:
        from lhotse.indexing import IndexedJsonlReader, LazyShuffledRange

        self.path = path
        self.shuffle = shuffle
        self.seed = seed
        self.index_path = index_path
        self._reader = IndexedJsonlReader(path, index_path=index_path)
        self._range = (
            LazyShuffledRange(len(self._reader), seed=seed) if shuffle else None
        )
        self._position = 0
        self._restored = False

    @property
    def is_indexed(self) -> bool:
        return True

    @property
    def has_constant_time_access(self) -> bool:
        return True

    def __getitem__(self, idx: int) -> Any:
        """O(1) random access: deserializes the *idx*-th item."""
        item = deserialize_item(self._reader[idx])
        return attach_graph_origin(item, idx)

    def __iter__(self):
        if self._restored:
            self._restored = False
            start = self._position
        else:
            start = 0
            if self._range is not None:
                self._range.reset()
        self._position = start

        n = len(self._reader)
        if self._range is not None:
            for i in range(start, n):
                self._position = i + 1
                phys_idx = self._range[i]
                item = deserialize_item(self._reader[phys_idx])
                attach_graph_origin(item, phys_idx)
                yield item
        else:
            for i in range(start, n):
                self._position = i + 1
                item = deserialize_item(self._reader[i])
                attach_graph_origin(item, i)
                yield item

    def __len__(self) -> int:
        return len(self._reader)

    def __add__(self, other) -> "LazyIteratorChain":
        return LazyIteratorChain(self, other)

    def state_dict(self) -> dict:
        sd = {"position": self._position, "shuffle": self.shuffle, "seed": self.seed}
        if self._range is not None:
            sd["range"] = self._range.state_dict()
        return sd

    def load_state_dict(self, sd: dict) -> None:
        self._position = sd["position"]
        if self._range is not None:
            if "range" not in sd:
                raise ValueError(
                    "LazyIndexedManifestIterator with shuffle=True requires "
                    "'range' in state_dict, but it was not found. "
                    "The checkpoint may have been created without shuffling."
                )
            self._range.load_state_dict(sd["range"])
        self._restored = True


class LazyIteratorChain(IteratorNode):
    """
    A thin wrapper over multiple iterators that enables to combine lazy manifests
    in Lhotse. It iterates all underlying iterables sequentially.

    It also supports shuffling via ``shuffle_iters``.  The shuffling strategy
    is chosen automatically based on whether all sub-iterators are indexed:

    * **Non-indexed sources** — shuffles the *order of sub-iterators* (shard-level
      shuffling).  Every iteration increments an internal counter so the shard
      order is re-randomized.
    * **Indexed sources** — uses a Feistel-cipher permutation over the combined
      index range for true *item-level* shuffling that crosses sub-iterator
      boundaries, via O(1) random access.

    Supports checkpointing via :meth:`state_dict` / :meth:`load_state_dict`.

    .. note:: if any of the input iterables is a dict, we'll iterate only its values.
    """

    is_checkpointable = True

    def __init__(
        self,
        *iterators: Iterable,
        shuffle_iters: bool = False,
        seed: Optional[Union[int, Literal["trng", "randomized"]]] = None,
    ) -> None:
        self.sources = []
        self.shuffle_iters = shuffle_iters
        self.seed = seed
        self.num_iters = 0
        for it in iterators:
            it = resolve_iterator_source(it)
            # Auto-flatten LazyIteratorChain instances if any are passed
            if isinstance(it, LazyIteratorChain):
                for sub_it in it.sources:
                    self.sources.append(sub_it)
            else:
                self.sources.append(it)
        # Iteration tracking (sequential path)
        self._current_iter_idx = 0
        self._iter_order: Optional[list] = None
        self._restored = False
        # Iteration tracking (globally-shuffled path)
        self._global_position = 0
        self._global_seed = None

    @property
    def is_indexed(self) -> bool:
        return all(getattr(s, "is_indexed", False) for s in self.sources)

    @property
    def has_constant_time_access(self) -> bool:
        if self.shuffle_iters and not self.is_indexed:
            return False  # shard order changes per iteration
        return all(supports_graph_restore(s, require_length=True) for s in self.sources)

    def __getitem__(self, idx: Any) -> Any:
        idx = normalize_graph_token(idx)
        if isinstance(idx, tuple) and len(idx) == 2:
            src_idx, source_token = idx
            return attach_graph_origin(self.sources[src_idx][source_token], idx)
        from bisect import bisect_right

        cum = self._cumulative_lengths()
        total = cum[-1]
        if idx < 0:
            idx += total
        if idx < 0 or idx >= total:
            raise IndexError("index out of range for LazyIteratorChain")
        src_idx = bisect_right(cum, idx)
        offset = idx - cum[src_idx - 1] if src_idx > 0 else idx
        return attach_graph_origin(self.sources[src_idx][offset], idx)

    def _cumulative_lengths(self) -> list:
        if getattr(self, "_cum_lens", None) is None:
            self._cum_lens = []
            total = 0
            for s in self.sources:
                total += len(s)
                self._cum_lens.append(total)
        return self._cum_lens

    def __iter__(self):
        if self.shuffle_iters and self.is_indexed:
            return self._iter_globally_shuffled()
        return self._iter_sequential()

    # ------------------------------------------------------------------
    # Sequential iteration (original path — with optional shard shuffle)
    # ------------------------------------------------------------------

    def _iter_sequential(self):
        from lhotse.dataset.dataloading import resolve_seed

        if self._restored:
            self._restored = False
            # Restore exact shard order and skip to the current shard.
            start_idx = self._current_iter_idx
            order = self._iter_order
            if order is None or len(order) != len(self.sources):
                order = list(range(len(self.sources)))
        else:
            start_idx = 0
            order = list(range(len(self.sources)))
            if self.shuffle_iters:
                if self.seed is None:
                    rng = random  # global Python RNG
                else:
                    rng = random.Random(resolve_seed(self.seed) + self.num_iters)
                rng.shuffle(order)
                self.num_iters += 1
            self._iter_order = order
            self._current_iter_idx = 0
        self._iter_order = order
        cum = self._cumulative_lengths()
        for idx in range(start_idx, len(order)):
            src_idx = order[idx]
            it = self.sources[src_idx]
            self._current_iter_idx = idx
            if isinstance(it, dict):
                it = it.values()
            for item in it:
                if self.has_constant_time_access and not self.shuffle_iters:
                    maybe_attach_graph_origin(item, (src_idx, get_graph_origin(item)))
                yield item

    # ------------------------------------------------------------------
    # Globally-shuffled iteration (O(1) random access across all sources)
    # ------------------------------------------------------------------

    def _iter_globally_shuffled(self):
        from lhotse.indexing import LazyShuffledRange

        total = len(self)
        if self._restored:
            self._restored = False
            start = self._global_position
            base_seed = self._global_seed
            if base_seed is None:
                base_seed = resolve_iteration_seed(self.seed)
        else:
            start = 0
            self._global_position = 0
            base_seed = resolve_iteration_seed(self.seed)
            self._global_seed = base_seed

        shuffled = LazyShuffledRange(total, seed=base_seed + self.num_iters)
        for i in range(start, total):
            self._global_position = i + 1
            yield self[shuffled[i]]
        self.num_iters += 1

    def __len__(self) -> int:
        return sum(len(it) for it in self.sources)

    def __add__(self, other) -> "LazyIteratorChain":
        return LazyIteratorChain(self, other)

    def state_dict(self) -> dict:
        sd = {
            "current_iter_idx": self._current_iter_idx,
            "num_iters": self.num_iters,
            "iter_order": self._iter_order,
            "global_position": self._global_position,
            "global_seed": getattr(self, "_global_seed", None),
        }
        # Save inner states for stateful children
        inner_states = []
        for s in self.sources:
            inner_states.append(_try_collect_child_state(s))
        sd["inner_states"] = inner_states
        return sd

    def load_state_dict(self, sd: dict) -> None:
        self._current_iter_idx = sd["current_iter_idx"]
        self.num_iters = sd["num_iters"]
        self._iter_order = sd.get("iter_order")
        self._global_position = sd.get("global_position", 0)
        self._global_seed = sd.get("global_seed")
        if self.shuffle_iters and self.is_indexed:
            # Globally-shuffled path: position + num_iters (+ stored per-epoch
            # resolved seed) are enough to reconstruct permutation deterministically.
            self._restored = True
            return
        # Sequential path: only restore sources that will still be iterated
        # (at or after current_iter_idx in iter_order).
        order = (
            self._iter_order
            if self._iter_order is not None
            else list(range(len(self.sources)))
        )
        active = set(order[self._current_iter_idx :])
        for i, (s, inner_sd) in enumerate(
            zip(self.sources, sd.get("inner_states", []))
        ):
            if i not in active or inner_sd is None:
                continue
            _try_restore_child_state(s, inner_sd)
        self._restored = True


class LazyIteratorMultiplexer(IteratorNode):
    """
    A wrapper over multiple iterators that enables to combine lazy manifests in Lhotse.
    During iteration, unlike :class:`.LazyIteratorChain`, :class:`.LazyIteratorMultiplexer`
    at each step randomly selects the iterable used to yield an item.

    Since the iterables might be of different length, we provide a ``weights`` parameter
    to let the user decide which iterables should be sampled more frequently than others.
    When an iterable is exhausted, we will keep sampling from the other iterables, until
    we exhaust them all, unless ``stop_early`` is set to ``True``.

    Supports checkpointing via :meth:`state_dict` / :meth:`load_state_dict`.
    """

    is_checkpointable = True

    def __init__(
        self,
        *iterators: Iterable,
        stop_early: bool = False,
        weights: Optional[List[Union[int, float]]] = None,
        seed: Union[int, Literal["trng", "randomized"]] = 0,
    ) -> None:
        self.sources = [resolve_iterator_source(it) for it in iterators]
        self.stop_early = stop_early
        self.seed = seed

        assert (
            len(self.sources) > 1
        ), "There have to be at least two iterables to multiplex."

        if weights is None:
            self.weights = [1] * len(self.sources)
        else:
            self.weights = weights

        assert len(self.sources) == len(self.weights)

        # Iteration state
        self._rng_state = None
        self._exhausted: Optional[list] = None
        self._restored = False

    @property
    def is_indexed(self) -> bool:
        return all(getattr(s, "is_indexed", False) for s in self.sources)

    @property
    def has_constant_time_access(self) -> bool:
        return all(supports_graph_restore(s) for s in self.sources)

    def __getitem__(self, token: Any) -> Any:
        token = normalize_graph_token(token)
        if not isinstance(token, tuple) or len(token) != 2:
            raise TypeError(
                "LazyIteratorMultiplexer expects graph restore tokens shaped like "
                "(source_index, source_token)."
            )
        source_idx, source_token = token
        return attach_graph_origin(self.sources[source_idx][source_token], token)

    def __iter__(self):
        from lhotse.dataset.dataloading import resolve_seed

        rng = random.Random(resolve_seed(self.seed))
        iters = [iter(it) for it in self.sources]
        restored = self._restored

        if restored:
            self._restored = False
            exhausted = (
                list(self._exhausted)
                if self._exhausted is not None
                else [False] * len(iters)
            )
            if self._rng_state is not None:
                rng.setstate(self._rng_state)
        else:
            exhausted = [False for _ in range(len(iters))]
        self._exhausted = exhausted

        def should_continue():
            if self.stop_early:
                return not any(exhausted)
            else:
                return not all(exhausted)

        while should_continue():
            active_indexes, active_weights = zip(
                *[
                    (i, w)
                    for i, (is_exhausted, w) in enumerate(zip(exhausted, self.weights))
                    if not is_exhausted
                ]
            )
            idx = rng.choices(active_indexes, weights=active_weights, k=1)[0]
            self._rng_state = rng.getstate()
            selected = iters[idx]
            try:
                item = next(selected)
                graph_token = None
                if self.has_constant_time_access:
                    graph_token = require_graph_origin(
                        item, "LazyIteratorMultiplexer", "items"
                    )
                maybe_attach_graph_origin(
                    item, None if graph_token is None else (idx, graph_token)
                )
                yield item
            except StopIteration:
                exhausted[idx] = True
                continue

    def __len__(self) -> int:
        return sum(len(it) for it in self.sources)

    def __add__(self, other) -> "LazyIteratorChain":
        return LazyIteratorChain(self, other)

    def state_dict(self) -> dict:
        sd = {
            "rng_state": self._rng_state,
            "exhausted": list(self._exhausted) if self._exhausted is not None else None,
        }
        inner_states = []
        for s in self.sources:
            inner_states.append(_try_collect_child_state(s))
        sd["inner_states"] = inner_states
        return sd

    def load_state_dict(self, sd: dict) -> None:
        self._rng_state = sd["rng_state"]
        self._exhausted = sd["exhausted"]
        active = None
        if self._exhausted is not None:
            active = {i for i, exhausted in enumerate(self._exhausted) if not exhausted}
        for i, (s, inner_sd) in enumerate(
            zip(self.sources, sd.get("inner_states", []))
        ):
            if active is not None and i not in active:
                continue
            _try_restore_child_state(s, inner_sd)
        self._restored = True


class LazyInfiniteApproximateMultiplexer(IteratorNode):
    """
    A variant of :class:`.LazyIteratorMultiplexer` that allows to control the number of
    iterables that are simultaneously open.

    It is useful for large-scale data sets where opening multiple file handles in
    many processes leads to exhaustion of the operating system resources.

    If the data sets are sharded, it is recommended to pass each shard as a separate iterator
    when creating objects of this class. It is OK to assign a dataset-level weight to each shard
    (e.g., if a dataset has a weight of 0.5, assign weight 0.5 to each of its shards).

    There are several differences between this class and :class:`.LazyIteratorMultiplexer`:
    * Objects of this class are infinite iterators.
    * We hold a list of ``max_open_streams`` open iterators at any given time.
        This list is filled by sampling input iterators with replacement.

    These differences are necessary to guarantee the weighted sampling property.
    If we did not sample with replacement or make it infinite, we would simply
    exhaust highly-weighted iterators towards the beginning of each "epoch"
    and keep sampling only lowly-weighted iterators towards the end of each "epoch".

    .. note:: This class does **not** support checkpointing
        (``state_dict`` / ``load_state_dict``).  Its infinite, approximate
        nature with dynamically replaced streams makes exact restoration
        infeasible.  For resumable multiplexed iteration, use
        :class:`.LazyIteratorMultiplexer` with finite sources instead.
    """

    def __init__(
        self,
        *iterators: Iterable,
        stop_early: bool = False,
        weights: Optional[List[Union[int, float]]] = None,
        seed: Union[int, Literal["trng", "randomized"]] = 0,
        max_open_streams: Optional[int] = None,
    ) -> None:
        self.sources = [resolve_iterator_source(it) for it in iterators]
        self.stop_early = stop_early
        self.seed = seed
        self.max_open_streams = max_open_streams
        if max_open_streams is None or max_open_streams > len(self.sources):
            self.max_open_streams = len(self.sources)

        assert len(self.sources) > 0
        self.weights = weights
        if weights is None:
            self.weights = [1] * len(self.sources)
        assert len(self.sources) == len(self.weights)
        assert (
            self.max_open_streams is None or self.max_open_streams >= 1
        ), f"{self.max_open_streams=}"

    def __iter__(self):
        """
        Assumptions
        - we have N streams but can only open M at the time (M < N)
        - the streams are finite
        - each stream needs to be "short" to ensure the mux property
        - each stream may be interpreted as a shard belonging to some larger group of streams
          (e.g. multiple shards of a given dataset).
        """
        from lhotse.dataset.dataloading import resolve_seed

        rng = random.Random(resolve_seed(self.seed))

        def shuffled_streams():
            indexes = list(range(len(self.sources)))
            while True:
                selected = rng.choices(indexes, self.weights, k=1)[0]
                yield self.sources[selected], self.weights[selected], selected

        # Initialize an infinite sequence of finite streams.
        stream_source = shuffled_streams()

        # Sample the first M active streams to be multiplexed.
        active_streams = [None] * self.max_open_streams
        active_weights = [None] * self.max_open_streams
        stream_indexes = list(range(self.max_open_streams))

        def sample_new_stream_at(pos: int) -> None:
            sampled_stream, sampled_weight, _ = next(stream_source)
            active_streams[pos] = iter(sampled_stream)
            active_weights[pos] = sampled_weight

        for stream_pos in range(self.max_open_streams):
            sample_new_stream_at(stream_pos)

        # The actual multiplexing loop.
        while True:
            stream_pos = rng.choices(
                stream_indexes,
                weights=active_weights if sum(active_weights) > 0 else None,
                k=1,
            )[0]
            selected = active_streams[stream_pos]
            try:
                item = next(selected)
                yield item
            except StopIteration:
                sample_new_stream_at(stream_pos)
                item = next(active_streams[stream_pos])
                yield item


class LazyShuffler(IteratorNode):
    """
    A wrapper over an iterable that enables lazy shuffling.
    The shuffling algorithm is reservoir-sampling based.
    See :func:`lhotse.utils.streaming_shuffle` for details.

    With graph-restorable indexed sources, the shuffle buffer and RNG state can
    be checkpointed and restored exactly.
    """

    def __init__(
        self,
        iterator: Iterable,
        buffer_size: int = 10000,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.source = resolve_iterator_source(iterator)
        self.buffer_size = buffer_size
        self.rng = rng if rng is not None else random.Random(random.getrandbits(64))
        self._buffer = deque()
        self._startup = True
        self._source_exhausted = False
        self._restored = False

    @property
    def is_checkpointable(self) -> bool:
        return supports_graph_restore(self.source)

    @property
    def is_indexed(self) -> bool:
        return getattr(self.source, "is_indexed", False)

    @property
    def has_constant_time_access(self) -> bool:
        return supports_graph_restore(self.source)

    def __getitem__(self, token: Any) -> Any:
        token = normalize_graph_token(token)
        return attach_graph_origin(self.source[token], token)

    def _reset_iteration_state(self) -> None:
        self._buffer.clear()
        self._startup = True
        self._source_exhausted = False

    def _next_source_item(self, source_iter) -> Any:
        try:
            return next(source_iter)
        except StopIteration:
            self._source_exhausted = True
            return None

    def _maybe_fill_buffer(self, source_iter) -> None:
        if len(self._buffer) >= self.buffer_size:
            return
        item = self._next_source_item(source_iter)
        if item is not None:
            self._buffer.append(item)

    def _swap_with_buffer(self, sample: Any) -> Any:
        if not self._buffer:
            return sample
        swap_idx = self.rng.randint(0, len(self._buffer) - 1)
        sample, self._buffer[swap_idx] = self._buffer[swap_idx], sample
        return sample

    def _startup_phase(self, source_iter):
        while self._startup and not self._source_exhausted:
            sample = self._next_source_item(source_iter)
            if sample is None:
                break
            self._maybe_fill_buffer(source_iter)
            sample = self._swap_with_buffer(sample)
            if len(self._buffer) < self.buffer_size:
                self._buffer.append(sample)
                continue
            self._startup = False
            yield sample

    def _steady_state_phase(self, source_iter):
        while not self._source_exhausted:
            sample = self._next_source_item(source_iter)
            if sample is None:
                break
            self._maybe_fill_buffer(source_iter)
            yield self._swap_with_buffer(sample)

    def __iter__(self):
        source_iter = iter(self.source)
        if self._restored:
            self._restored = False
        else:
            self._reset_iteration_state()

        yield from self._startup_phase(source_iter)
        yield from self._steady_state_phase(source_iter)

        while self._buffer:
            yield self._buffer.popleft()

    def __len__(self) -> int:
        return len(self.source)

    def __add__(self, other) -> "LazyIteratorChain":
        return LazyIteratorChain(self, other)

    def state_dict(self) -> dict:
        if not self.is_checkpointable:
            raise NotImplementedError(
                "LazyShuffler does not support checkpointing unless its source "
                "supports graph restoration."
            )
        from lhotse.checkpoint import _rng_state_to_json

        source_state = _try_collect_child_state(self.source)
        return {
            "buffer": [
                require_graph_origin(item, "LazyShuffler", "buffered items")
                for item in self._buffer
            ],
            "startup": self._startup,
            "source_exhausted": self._source_exhausted,
            "rng_state": _rng_state_to_json(self.rng.getstate()),
            "source": source_state,
        }

    def load_state_dict(self, sd: dict) -> None:
        if not self.is_checkpointable:
            raise NotImplementedError(
                "LazyShuffler does not support checkpointing unless its source "
                "supports graph restoration."
            )
        from lhotse.checkpoint import _rng_state_from_json

        _try_restore_child_state(self.source, sd.get("source"))
        self._buffer = deque(
            self.source[normalize_graph_token(token)] for token in sd.get("buffer", [])
        )
        self._startup = sd.get("startup", True)
        self._source_exhausted = sd.get("source_exhausted", False)
        self.rng.setstate(_rng_state_from_json(sd["rng_state"]))
        self._restored = True


class LazyFilter(IteratorNode):
    """
    A wrapper over an iterable that enables lazy filtering.
    It works like Python's `filter` built-in by applying the filter predicate
    to each element and yielding it further if predicate returned ``True``.

    Supports checkpointing via :meth:`state_dict` / :meth:`load_state_dict`
    (delegates to the inner ``source`` iterator; the filter itself is stateless).
    """

    is_checkpointable = True

    def __init__(self, iterator: Iterable, predicate: Callable[[Any], bool]) -> None:
        self.source = resolve_iterator_source(iterator)
        self.predicate = predicate
        assert callable(
            self.predicate
        ), f"LazyFilter: 'predicate' arg must be callable (got {predicate})."
        if (
            isinstance(self.predicate, types.LambdaType)
            and self.predicate.__name__ == "<lambda>"
            and not is_module_available("dill")
        ):
            warnings.warn(
                "A lambda was passed to LazyFilter: it may prevent you from forking this process. "
                "If you experience issues with num_workers > 0 in torch.utils.data.DataLoader, "
                "try passing a regular function instead."
            )

    @property
    def is_indexed(self) -> bool:
        return getattr(self.source, "is_indexed", False)

    @property
    def has_constant_time_access(self) -> bool:
        return supports_graph_restore(self.source)

    def __getitem__(self, token: Any) -> Any:
        token = normalize_graph_token(token)
        item = self.source[token]
        if not self.predicate(item):
            raise RuntimeError(
                "LazyFilter received a graph restore token that does not satisfy its "
                "predicate."
            )
        return attach_graph_origin(item, token)

    def __iter__(self):
        for item in self.source:
            if self.predicate(item):
                yield item

    def __add__(self, other) -> "LazyIteratorChain":
        return LazyIteratorChain(self, other)

    def __len__(self) -> int:
        raise TypeError(
            "LazyFilter does not support __len__ because it would require "
            "iterating over the whole iterator, which is not possible in a lazy fashion. "
            "If you really need to know the length, convert to eager mode first using "
            "`.to_eager()`. Note that this will require loading the whole iterator into memory."
        )

    def state_dict(self) -> dict:
        sd = {}
        source_state = _try_collect_child_state(self.source)
        if source_state is not None:
            sd["source"] = source_state
        return sd

    def load_state_dict(self, sd: dict) -> None:
        _try_restore_child_state(self.source, sd.get("source"))


class LazyMapper(IteratorNode):
    """
    A wrapper over an iterable that enables lazy function evaluation on each item.
    It works like Python's `map` built-in by applying a callable ``fn``
    to each element ``x`` and yielding the result of ``fn(x)`` further.

    New in Lhotse v1.22.0: ``apply_fn`` can be provided to decide whether ``fn`` should be applied
        to a given example or not (in which case it will return it as-is, i.e., it does not filter).

    Supports checkpointing via :meth:`state_dict` / :meth:`load_state_dict`
    (delegates to the inner ``source`` iterator; the mapper itself is stateless).
    """

    is_checkpointable = True

    def __init__(
        self,
        iterator: Iterable,
        fn: Callable[[Any], Any],
        apply_fn: Optional[Callable[[Any], bool]] = None,
    ) -> None:
        self.source = resolve_iterator_source(iterator)
        self.fn = fn
        self.apply_fn = apply_fn
        assert callable(self.fn), f"LazyMapper: 'fn' arg must be callable (got {fn})."
        if self.apply_fn is not None:
            assert callable(
                self.apply_fn
            ), f"LazyMapper: 'apply_fn' arg must be callable (got {fn})."
        if (
            (isinstance(self.fn, types.LambdaType) and self.fn.__name__ == "<lambda>")
            or (
                isinstance(self.apply_fn, types.LambdaType)
                and self.apply_fn.__name__ == "<lambda>"
            )
            and not is_dill_enabled()
        ):
            warnings.warn(
                "A lambda was passed to LazyMapper: it may prevent you from forking this process. "
                "If you experience issues with num_workers > 0 in torch.utils.data.DataLoader, "
                "try passing a regular function instead."
            )

    @property
    def is_indexed(self) -> bool:
        return getattr(self.source, "is_indexed", False)

    @property
    def has_constant_time_access(self) -> bool:
        return supports_graph_restore(self.source)

    def __getitem__(self, idx: Any) -> Any:
        graph_token = normalize_graph_token(idx)
        item = self.source[graph_token]
        if self.apply_fn is None or self.apply_fn(item):
            item = self.fn(item)
        return attach_graph_origin(item, graph_token)

    def __iter__(self):
        if self.apply_fn is None:
            for item in self.source:
                graph_idx = get_graph_origin(item)
                item = self.fn(item)
                yield maybe_attach_graph_origin(item, graph_idx)
        else:
            for item in self.source:
                graph_idx = get_graph_origin(item)
                if self.apply_fn(item):
                    ans = self.fn(item)
                else:
                    ans = item
                yield maybe_attach_graph_origin(ans, graph_idx)

    def __len__(self) -> int:
        return len(self.source)

    def __add__(self, other) -> "LazyIteratorChain":
        return LazyIteratorChain(self, other)

    def state_dict(self) -> dict:
        sd = {}
        source_state = _try_collect_child_state(self.source)
        if source_state is not None:
            sd["source"] = source_state
        return sd

    def load_state_dict(self, sd: dict) -> None:
        _try_restore_child_state(self.source, sd.get("source"))


class LazyFlattener(IteratorNode):
    """
    A wrapper over an iterable of collections that flattens it to an iterable of items.

    With graph-restorable outer sources, this node checkpoints exactly by saving
    the current outer-item token and the local offset within that collection.

    Example::

        >>> list_of_cut_sets: List[CutSet] = [CutSet(...), CutSet(...)]
        >>> list_of_cuts: List[Cut] = list(LazyFlattener(list_of_cut_sets))
    """

    def __init__(self, iterator: Iterable) -> None:
        self.source = resolve_iterator_source(iterator)
        self._active_outer_token = None
        self._inner_position = 0
        self._restored = False

    @property
    def is_checkpointable(self) -> bool:
        return supports_graph_restore(self.source)

    @property
    def is_indexed(self) -> bool:
        return getattr(self.source, "is_indexed", False)

    @property
    def has_constant_time_access(self) -> bool:
        return supports_graph_restore(self.source)

    def _resolve_collection(self, collection: Any) -> Any:
        return resolve_iterator_source(collection)

    def _inner_token(self, item: Any, inner_idx: int) -> Any:
        token = get_graph_origin(item)
        return inner_idx if token is None else token

    def _restore_inner_item(self, collection: Any, token: Any) -> Any:
        collection = self._resolve_collection(collection)
        token = normalize_graph_token(token)
        if isinstance(token, int):
            if hasattr(collection, "__getitem__"):
                return collection[token]
            for idx, item in enumerate(collection):
                if idx == token:
                    return item
            raise IndexError(
                f"LazyFlattener inner index {token} is out of range for {type(collection).__name__}."
            )
        if supports_graph_restore(collection):
            return collection[token]
        raise RuntimeError(
            "LazyFlattener received a non-integer inner graph token for a collection "
            "that does not support graph restoration."
        )

    def __getitem__(self, idx: Any) -> Any:
        token = normalize_graph_token(idx)
        if not isinstance(token, tuple) or len(token) != 2:
            raise TypeError(
                "LazyFlattener expects graph restore tokens shaped like "
                "(outer_token, inner_token)."
            )
        outer_token, inner_token = token
        collection = self.source[outer_token]
        item = self._restore_inner_item(collection, inner_token)
        return attach_graph_origin(item, token)

    def _iter_collection(
        self, collection: Any, outer_token: Any, start_inner: int = 0
    ) -> Iterable[Any]:
        collection = self._resolve_collection(collection)
        for inner_idx, item in enumerate(collection):
            if inner_idx < start_inner:
                continue
            self._active_outer_token = outer_token
            self._inner_position = inner_idx + 1
            token = None
            if outer_token is not None:
                token = (outer_token, self._inner_token(item, inner_idx))
            yield maybe_attach_graph_origin(item, token)
        self._active_outer_token = None
        self._inner_position = 0

    def __iter__(self):
        if self._restored and self._active_outer_token is not None:
            collection = self.source[self._active_outer_token]
            yield from self._iter_collection(
                collection,
                self._active_outer_token,
                start_inner=self._inner_position,
            )
        self._restored = False
        for cuts in self.source:
            outer_token = (
                require_graph_origin(cuts, "LazyFlattener", "outer collections")
                if self.is_checkpointable
                else None
            )
            yield from self._iter_collection(cuts, outer_token)

    def __add__(self, other) -> "LazyIteratorChain":
        return LazyIteratorChain(self, other)

    def __len__(self) -> int:
        raise TypeError(
            "LazyFlattener does not support __len__ because it would require "
            "iterating over the whole iterator, which is not possible in a lazy fashion. "
            "If you really need to know the length, convert to eager mode first using "
            "`.to_eager()`. Note that this will require loading the whole iterator into memory."
        )

    def state_dict(self) -> dict:
        if not self.is_checkpointable:
            raise NotImplementedError(
                "LazyFlattener does not support checkpointing unless its outer "
                "source supports graph restoration."
            )
        source_state = _try_collect_child_state(self.source)
        state = {
            "active_outer_token": self._active_outer_token,
            "inner_position": self._inner_position,
            "source": source_state,
        }
        return state

    def load_state_dict(self, sd: dict) -> None:
        if not self.is_checkpointable:
            raise NotImplementedError(
                "LazyFlattener does not support checkpointing unless its outer "
                "source supports graph restoration."
            )
        self._active_outer_token = normalize_graph_token(sd.get("active_outer_token"))
        self._inner_position = sd.get("inner_position", 0)
        _try_restore_child_state(self.source, sd.get("source"))
        self._restored = True


class LazyRepeater(IteratorNode):
    """
    A wrapper over an iterable that enables to repeat it N times or infinitely (default).

    Supports checkpointing via :meth:`state_dict` / :meth:`load_state_dict`.
    Captures the current epoch and the state of the inner ``source`` iterator.
    """

    is_checkpointable = True

    def __init__(
        self, iterator: Iterable, times: Optional[int] = None, preserve_id: bool = False
    ) -> None:
        self.source = resolve_iterator_source(iterator)
        self.times = times
        self.preserve_id = preserve_id
        assert self.times is None or self.times > 0
        self._current_epoch = 0
        self._restored = False

    @property
    def is_indexed(self) -> bool:
        return getattr(self.source, "is_indexed", False)

    @property
    def has_constant_time_access(self) -> bool:
        return supports_graph_restore(self.source)

    def __getitem__(self, idx: Any) -> Any:
        graph_token = normalize_graph_token(idx)
        if isinstance(graph_token, tuple) and len(graph_token) == 2:
            repeat_idx, source_token = graph_token
            item = self.source[source_token]
        else:
            n = len(self.source)
            repeat_idx = graph_token // n
            item = self.source[graph_token % n]
        if self.preserve_id:
            return attach_graph_origin(item, graph_token)
        return attach_graph_origin(
            attach_repeat_idx_to_id(item, repeat_idx), graph_token
        )

    def __iter__(self):
        restored = self._restored
        epoch = self._current_epoch if restored else 0
        self._restored = False
        while self.times is None or epoch < self.times:
            self._current_epoch = epoch
            if self.preserve_id:
                iterator = self.source
            else:
                iterator = LazyMapper(
                    self.source, partial(attach_repeat_idx_to_id, idx=epoch)
                )
            at_least_once = False
            for item in iterator:
                at_least_once = True
                source_idx = get_graph_origin(item)
                maybe_attach_graph_origin(
                    item, None if source_idx is None else (epoch, source_idx)
                )
                yield item
            if not at_least_once and not restored:
                return  # Detect empty iterables to avoid hanging the program.
            # After the first (possibly restored) epoch, behave normally.
            restored = False
            epoch += 1

    def __len__(self) -> int:
        if self.times is None:
            raise TypeError(
                f"object of type '{type(self).__name__}' is an infinite iterator"
            )
        return len(self.source) * self.times

    def __add__(self, other) -> "LazyIteratorChain":
        return LazyIteratorChain(self, other)

    def state_dict(self) -> dict:
        sd = {"current_epoch": self._current_epoch}
        source_state = _try_collect_child_state(self.source)
        if source_state is not None:
            sd["source"] = source_state
        return sd

    def load_state_dict(self, sd: dict) -> None:
        self._current_epoch = sd["current_epoch"]
        _try_restore_child_state(self.source, sd.get("source"))
        self._restored = True


class LazySlicer(IteratorNode):
    """
    A wrapper over an iterable that enables selecting k-th element every n elements.

    Supports checkpointing via :meth:`state_dict` / :meth:`load_state_dict`
    (delegates to the inner ``source`` iterator).
    """

    is_checkpointable = True

    def __init__(self, iterator: Iterable, k: int, n: int) -> None:
        self.source = resolve_iterator_source(iterator)
        assert (
            k < n
        ), f"When selecting k-th element every n elements, k must be less than n (got k={k} n={n})."
        self.k = k
        self.n = n
        self._source_offset = 0
        self._restored = False

    @property
    def is_indexed(self) -> bool:
        return getattr(self.source, "is_indexed", False)

    @property
    def has_constant_time_access(self) -> bool:
        return supports_graph_restore(self.source)

    def __getitem__(self, idx: Any) -> Any:
        graph_token = normalize_graph_token(idx)
        if (
            isinstance(graph_token, tuple)
            and len(graph_token) == 2
            and graph_token[0] == "source"
        ):
            return attach_graph_origin(self.source[graph_token[1]], graph_token)
        if isinstance(graph_token, int):
            return attach_graph_origin(self.source[graph_token * self.n + self.k], idx)
        return attach_graph_origin(self.source[graph_token], graph_token)

    def __iter__(self):
        start = self._source_offset if self._restored else 0
        self._restored = False
        for idx, item in enumerate(self.source, start=start):
            self._source_offset = idx + 1
            if idx % self.n == self.k:
                source_idx = get_graph_origin(item)
                maybe_attach_graph_origin(
                    item, None if source_idx is None else ("source", source_idx)
                )
                yield item

    def __add__(self, other) -> "LazyIteratorChain":
        return LazyIteratorChain(self, other)

    def __len__(self) -> int:
        raise TypeError(
            "LazySlicer does not support __len__ because it would require "
            "iterating over the whole iterator, which is not possible in a lazy fashion. "
            "If you really need to know the length, convert to eager mode first using "
            "`.to_eager()`. Note that this will require loading the whole iterator into memory."
        )

    def state_dict(self) -> dict:
        sd = {"source_offset": self._source_offset}
        source_state = _try_collect_child_state(self.source)
        if source_state is not None:
            sd["source"] = source_state
        return sd

    def load_state_dict(self, sd: dict) -> None:
        self._source_offset = sd.get("source_offset", 0)
        _try_restore_child_state(self.source, sd.get("source"))
        self._restored = True


def attach_repeat_idx_to_id(item: Any, idx: int) -> Any:
    if not hasattr(item, "id"):
        return item
    return fastcopy(item, id=f"{item.id}_repeat{idx}")


def count_newlines_fast(path: Pathlike):
    """
    Counts newlines in a file using buffered chunk reads.
    The fastest possible option in Python according to:
    https://stackoverflow.com/a/68385697/5285891
    (This is a slightly modified variant of that answer.)
    """

    def _make_gen(reader):
        b = reader(2**16)
        while b:
            yield b
            b = reader(2**16)

    read_mode = "rb" if not str(path) == "-" else "r"
    with open_best(path, read_mode) as f:
        count = sum(buf.count(b"\n") for buf in _make_gen(f.read))
    return count
