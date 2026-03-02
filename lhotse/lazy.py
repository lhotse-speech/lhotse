import os
import random
import types
import warnings
from contextlib import contextmanager
from functools import partial
from typing import Any, Callable, Iterable, List, Literal, Optional, TypeVar, Union

from lhotse.serialization import (
    LazyMixin,
    decode_json_line,
    deserialize_item,
    open_best,
)
from lhotse.utils import Pathlike, fastcopy, is_module_available, streaming_shuffle

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Stateful iterator protocol
# ---------------------------------------------------------------------------


class StatefulIterator:
    """
    Protocol (mixin) for lazy iterators that support checkpointing.

    Subclasses must implement :meth:`state_dict` and :meth:`load_state_dict`.
    The convention for child references is:

    * ``self.source``  — single child iterator
    * ``self.sources`` — list of child iterators
    """

    def state_dict(self) -> dict:
        raise NotImplementedError

    def load_state_dict(self, sd: dict) -> None:
        raise NotImplementedError


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
            return cls(LazyFilter(self, predicate=predicate))

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
        ans = cls(LazyMapper(self.data, fn=transform_fn))
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
            return cls(LazyShuffler(self.data, buffer_size=buffer_size, rng=rng))
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
        return cls(LazyRepeater(self, times=times, preserve_id=preserve_id))

    def __add__(self, other):
        cls = type(self)
        return cls(LazyIteratorChain(self.data, other.data))


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


class LazyTxtIterator:
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


class LazyJsonlIterator(StatefulIterator):
    """
    LazyJsonlIterator provides the ability to read JSON lines as Python dicts.
    It can also provide the number of lines via __len__ via fast newlines counting.

    Supports checkpointing via :meth:`state_dict` / :meth:`load_state_dict`.
    When a binary index file (``.idx``) is available, restoration jumps directly
    to the saved position in O(1).  Otherwise a sequential fast-forward is used.
    """

    def __init__(self, path: Pathlike) -> None:
        self.path = path
        self._len = None
        self._position = 0
        self._restored = False

    def __iter__(self):
        from lhotse.indexing import IndexedJsonlReader, index_exists

        start = self._position if self._restored else 0
        self._restored = False
        self._position = start
        if index_exists(self.path):
            reader = IndexedJsonlReader(self.path)
            for i in range(start, len(reader)):
                self._position = i + 1
                yield reader[i]
        else:
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


class LazyManifestIterator(Dillable, StatefulIterator):
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


class LazyIteratorChain(Dillable, StatefulIterator):
    """
    A thin wrapper over multiple iterators that enables to combine lazy manifests
    in Lhotse. It iterates all underlying iterables sequentially.

    It also supports shuffling the sub-iterators when it's iterated over.
    This can be used to implement sharding (where each iterator is a shard)
    with randomized shard order. Every iteration of this object will increment
    an internal counter so that the next time it's iterated, the order of shards
    is again randomized.

    Supports checkpointing via :meth:`state_dict` / :meth:`load_state_dict`.

    .. note:: if any of the input iterables is a dict, we'll iterate only its values.
    """

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
            # Auto-flatten LazyIteratorChain instances if any are passed
            if isinstance(it, LazyIteratorChain):
                for sub_it in it.sources:
                    self.sources.append(sub_it)
            else:
                self.sources.append(it)
        # Iteration tracking
        self._current_iter_idx = 0
        self._iter_order: Optional[list] = None
        self._restored = False

    def __iter__(self):
        from lhotse.dataset.dataloading import resolve_seed

        if self._restored:
            self._restored = False
            # Restore exact shard order and skip to the current shard
            start_idx = self._current_iter_idx
            sources = [self.sources[i] for i in self._iter_order]
        else:
            start_idx = 0
            sources = list(self.sources)  # shallow copy for potential shuffle
            if self.shuffle_iters:
                if self.seed is None:
                    rng = random  # global Python RNG
                else:
                    rng = random.Random(resolve_seed(self.seed) + self.num_iters)
                rng.shuffle(sources)
                self.num_iters += 1
            self._iter_order = [self.sources.index(s) for s in sources]
            self._current_iter_idx = 0
        for idx, it in enumerate(sources):
            if idx < start_idx:
                continue
            self._current_iter_idx = idx
            if isinstance(it, dict):
                it = it.values()
            yield from it

    def __len__(self) -> int:
        return sum(len(it) for it in self.sources)

    def __add__(self, other) -> "LazyIteratorChain":
        return LazyIteratorChain(self, other)

    def state_dict(self) -> dict:
        sd = {
            "current_iter_idx": self._current_iter_idx,
            "num_iters": self.num_iters,
            "iter_order": self._iter_order,
        }
        # Save inner states for stateful children
        inner_states = []
        for s in self.sources:
            if isinstance(s, StatefulIterator):
                inner_states.append(s.state_dict())
            else:
                inner_states.append(None)
        sd["inner_states"] = inner_states
        return sd

    def load_state_dict(self, sd: dict) -> None:
        self._current_iter_idx = sd["current_iter_idx"]
        self.num_iters = sd["num_iters"]
        self._iter_order = sd["iter_order"]
        # Only restore sources that will still be iterated (at or after
        # current_iter_idx in iter_order).  Sources before that index were
        # fully consumed and should start fresh on the next epoch.
        order = (
            self._iter_order
            if self._iter_order is not None
            else list(range(len(self.sources)))
        )
        active = set(order[self._current_iter_idx :])
        for i, (s, inner_sd) in enumerate(
            zip(self.sources, sd.get("inner_states", []))
        ):
            if i in active and inner_sd is not None and isinstance(s, StatefulIterator):
                s.load_state_dict(inner_sd)
        self._restored = True


class LazyIteratorMultiplexer(Dillable, StatefulIterator):
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

    def __init__(
        self,
        *iterators: Iterable,
        stop_early: bool = False,
        weights: Optional[List[Union[int, float]]] = None,
        seed: Union[int, Literal["trng", "randomized"]] = 0,
    ) -> None:
        self.sources = list(iterators)
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

    def __iter__(self):
        from lhotse.dataset.dataloading import resolve_seed

        rng = random.Random(resolve_seed(self.seed))
        iters = [iter(it) for it in self.sources]

        if self._restored:
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
            if isinstance(s, StatefulIterator):
                inner_states.append(s.state_dict())
            else:
                inner_states.append(None)
        sd["inner_states"] = inner_states
        return sd

    def load_state_dict(self, sd: dict) -> None:
        self._rng_state = sd["rng_state"]
        self._exhausted = sd["exhausted"]
        for s, inner_sd in zip(self.sources, sd.get("inner_states", [])):
            if inner_sd is not None and isinstance(s, StatefulIterator):
                s.load_state_dict(inner_sd)
        self._restored = True


class LazyInfiniteApproximateMultiplexer(Dillable):
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
        self.sources = list(iterators)
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
        active_origins = [None] * self.max_open_streams
        stream_indexes = list(range(self.max_open_streams))

        def sample_new_stream_at(pos: int) -> None:
            sampled_stream, sampled_weight, origin_idx = next(stream_source)
            active_streams[pos] = iter(sampled_stream)
            active_weights[pos] = sampled_weight
            active_origins[pos] = origin_idx

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


class LazyShuffler(Dillable):
    """
    A wrapper over an iterable that enables lazy shuffling.
    The shuffling algorithm is reservoir-sampling based.
    See :func:`lhotse.utils.streaming_shuffle` for details.

    .. note:: LazyShuffler does **not** support checkpointing
        (``state_dict`` / ``load_state_dict``).  The internal shuffle buffer
        cannot be cheaply saved or reconstructed.  For resumable shuffled
        iteration, use indexed random-access reads instead.
    """

    def __init__(
        self,
        iterator: Iterable,
        buffer_size: int = 10000,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.source = iterator
        self.buffer_size = buffer_size
        self.rng = rng

    def __iter__(self):
        return iter(
            streaming_shuffle(
                iter(self.source),
                bufsize=self.buffer_size,
                rng=self.rng,
            )
        )

    def __len__(self) -> int:
        return len(self.source)

    def __add__(self, other) -> "LazyIteratorChain":
        return LazyIteratorChain(self, other)


class LazyFilter(Dillable, StatefulIterator):
    """
    A wrapper over an iterable that enables lazy filtering.
    It works like Python's `filter` built-in by applying the filter predicate
    to each element and yielding it further if predicate returned ``True``.

    Supports checkpointing via :meth:`state_dict` / :meth:`load_state_dict`
    (delegates to the inner ``source`` iterator; the filter itself is stateless).
    """

    def __init__(self, iterator: Iterable, predicate: Callable[[Any], bool]) -> None:
        self.source = iterator
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

    def __iter__(self):
        return filter(self.predicate, self.source)

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
        if isinstance(self.source, StatefulIterator):
            sd["source"] = self.source.state_dict()
        return sd

    def load_state_dict(self, sd: dict) -> None:
        if "source" in sd and isinstance(self.source, StatefulIterator):
            self.source.load_state_dict(sd["source"])


class LazyMapper(Dillable, StatefulIterator):
    """
    A wrapper over an iterable that enables lazy function evaluation on each item.
    It works like Python's `map` built-in by applying a callable ``fn``
    to each element ``x`` and yielding the result of ``fn(x)`` further.

    New in Lhotse v1.22.0: ``apply_fn`` can be provided to decide whether ``fn`` should be applied
        to a given example or not (in which case it will return it as-is, i.e., it does not filter).

    Supports checkpointing via :meth:`state_dict` / :meth:`load_state_dict`
    (delegates to the inner ``source`` iterator; the mapper itself is stateless).
    """

    def __init__(
        self,
        iterator: Iterable,
        fn: Callable[[Any], Any],
        apply_fn: Optional[Callable[[Any], bool]] = None,
    ) -> None:
        self.source = iterator
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

    def __iter__(self):
        if self.apply_fn is None:
            yield from map(self.fn, self.source)
        else:
            for item in self.source:
                if self.apply_fn(item):
                    ans = self.fn(item)
                else:
                    ans = item
                yield ans

    def __len__(self) -> int:
        return len(self.source)

    def __add__(self, other) -> "LazyIteratorChain":
        return LazyIteratorChain(self, other)

    def state_dict(self) -> dict:
        sd = {}
        if isinstance(self.source, StatefulIterator):
            sd["source"] = self.source.state_dict()
        return sd

    def load_state_dict(self, sd: dict) -> None:
        if "source" in sd and isinstance(self.source, StatefulIterator):
            self.source.load_state_dict(sd["source"])


class LazyFlattener(Dillable, StatefulIterator):
    """
    A wrapper over an iterable of collections that flattens it to an iterable of items.

    Supports checkpointing via :meth:`state_dict` / :meth:`load_state_dict`
    (delegates to the inner ``source`` iterator).

    Example::

        >>> list_of_cut_sets: List[CutSet] = [CutSet(...), CutSet(...)]
        >>> list_of_cuts: List[Cut] = list(LazyFlattener(list_of_cut_sets))
    """

    def __init__(self, iterator: Iterable) -> None:
        self.source = iterator

    def __iter__(self):
        for cuts in self.source:
            yield from cuts

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
        sd = {}
        if isinstance(self.source, StatefulIterator):
            sd["source"] = self.source.state_dict()
        return sd

    def load_state_dict(self, sd: dict) -> None:
        if "source" in sd and isinstance(self.source, StatefulIterator):
            self.source.load_state_dict(sd["source"])


class LazyRepeater(Dillable, StatefulIterator):
    """
    A wrapper over an iterable that enables to repeat it N times or infinitely (default).

    Supports checkpointing via :meth:`state_dict` / :meth:`load_state_dict`.
    Captures the current epoch and the state of the inner ``source`` iterator.
    """

    def __init__(
        self, iterator: Iterable, times: Optional[int] = None, preserve_id: bool = False
    ) -> None:
        self.source = iterator
        self.times = times
        self.preserve_id = preserve_id
        assert self.times is None or self.times > 0
        self._current_epoch = 0
        self._restored = False

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
        if isinstance(self.source, StatefulIterator):
            sd["source"] = self.source.state_dict()
        return sd

    def load_state_dict(self, sd: dict) -> None:
        self._current_epoch = sd["current_epoch"]
        if "source" in sd and isinstance(self.source, StatefulIterator):
            self.source.load_state_dict(sd["source"])
        self._restored = True


class LazySlicer(Dillable, StatefulIterator):
    """
    A wrapper over an iterable that enables selecting k-th element every n elements.

    Supports checkpointing via :meth:`state_dict` / :meth:`load_state_dict`
    (delegates to the inner ``source`` iterator).
    """

    def __init__(self, iterator: Iterable, k: int, n: int) -> None:
        self.source = iterator
        assert (
            k < n
        ), f"When selecting k-th element every n elements, k must be less than n (got k={k} n={n})."
        self.k = k
        self.n = n
        self._source_offset = 0
        self._restored = False

    def __iter__(self):
        start = self._source_offset if self._restored else 0
        self._restored = False
        for idx, item in enumerate(self.source, start=start):
            self._source_offset = idx + 1
            if idx % self.n == self.k:
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
        if isinstance(self.source, StatefulIterator):
            sd["source"] = self.source.state_dict()
        return sd

    def load_state_dict(self, sd: dict) -> None:
        self._source_offset = sd.get("source_offset", 0)
        if "source" in sd and isinstance(self.source, StatefulIterator):
            self.source.load_state_dict(sd["source"])
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
