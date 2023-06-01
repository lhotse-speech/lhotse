import random
import types
import warnings
from functools import partial
from typing import Any, Callable, Iterable, List, Optional, TypeVar, Union

from lhotse.serialization import (
    LazyMixin,
    decode_json_line,
    deserialize_item,
    extension_contains,
    open_best,
)
from lhotse.utils import Pathlike, fastcopy, is_module_available, streaming_shuffle

T = TypeVar("T")


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

        if self.is_lazy:
            return cls(LazyMapper(self.data, fn=transform_fn))

        return cls.from_items(transform_fn(item) for item in self)

    @classmethod
    def mux(
        cls,
        *manifests,
        stop_early: bool = False,
        weights: Optional[List[Union[int, float]]] = None,
        seed: int = 0,
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
            ids = list(self.ids)
            rng.shuffle(ids)
            return cls({id_: self[id_] for id_ in ids})

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

    def __getstate__(self):
        if is_module_available("dill"):
            import dill

            return dill.dumps(self.__dict__)
        else:
            return self.__dict__

    def __setstate__(self, state):
        if is_module_available("dill"):
            import dill

            self.__dict__ = dill.loads(state)
        else:
            self.__dict__ = state


class ImitatesDict(Dillable):
    """
    Helper base class for lazy iterators defined below.
    It exists to make them drop-in replacements for data-holding dicts
    in Lhotse's CutSet, RecordingSet, etc. classes.
    """

    def __iter__(self):
        raise NotImplemented

    def values(self):
        yield from self

    def keys(self):
        return (item.id for item in self)

    def items(self):
        return ((item.id, item) for item in self)


class LazyJsonlIterator:
    """
    LazyJsonlIterator provides the ability to read JSON lines as Python dicts.
    It can also provide the number of lines via __len__ via fast newlines counting.
    """

    def __init__(self, path: Pathlike) -> None:
        self.path = path
        self._len = None

    def __iter__(self):
        with open_best(self.path, "r") as f:
            for line in f:
                data = decode_json_line(line)
                yield data

    def __len__(self) -> int:
        if self._len is None:
            self._len = count_newlines_fast(self.path)
        return self._len


class LazyManifestIterator(ImitatesDict):
    """
    LazyManifestIterator provides the ability to read Lhotse objects from a
    JSONL file on-the-fly, without reading its full contents into memory.

    This class is designed to be a partial "drop-in" replacement for ordinary dicts
    to support lazy loading of RecordingSet, SupervisionSet and CutSet.
    Since it does not support random access reads, some methods of these classes
    might not work properly.
    """

    def __init__(self, path: Pathlike) -> None:
        assert extension_contains(".jsonl", path) or str(path) == "-"
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


class LazyIteratorChain(ImitatesDict):
    """
    A thin wrapper over multiple iterators that enables to combine lazy manifests
    in Lhotse. It iterates all underlying iterables sequentially.

    It also supports shuffling the sub-iterators when it's iterated over.
    This can be used to implement sharding (where each iterator is a shard)
    with randomized shard order. Every iteration of this object will increment
    an internal counter so that the next time it's iterated, the order of shards
    is again randomized.

    .. note:: if any of the input iterables is a dict, we'll iterate only its values.
    """

    def __init__(
        self,
        *iterators: Iterable,
        shuffle_iters: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.iterators = []
        self.shuffle_iters = shuffle_iters
        self.seed = seed
        self.num_iters = 0
        for it in iterators:
            # Auto-flatten LazyIteratorChain instances if any are passed
            if isinstance(it, LazyIteratorChain):
                for sub_it in it.iterators:
                    self.iterators.append(sub_it)
            else:
                self.iterators.append(it)

    def __iter__(self):
        iterators = self.iterators
        if self.shuffle_iters:
            if self.seed is None:
                rng = random  # global Python RNG
            else:
                rng = random.Random(self.seed + self.num_iters)
            rng.shuffle(iterators)
            self.num_iters += 1
        for it in iterators:
            if isinstance(it, dict):
                it = it.values()
            yield from it

    def __len__(self) -> int:
        return sum(len(it) for it in self.iterators)

    def __add__(self, other) -> "LazyIteratorChain":
        return LazyIteratorChain(self, other)


class LazyIteratorMultiplexer(ImitatesDict):
    """
    A wrapper over multiple iterators that enables to combine lazy manifests in Lhotse.
    During iteration, unlike :class:`.LazyIteratorChain`, :class:`.LazyIteratorMultiplexer`
    at each step randomly selects the iterable used to yield an item.

    Since the iterables might be of different length, we provide a ``weights`` parameter
    to let the user decide which iterables should be sampled more frequently than others.
    When an iterable is exhausted, we will keep sampling from the other iterables, until
    we exhaust them all, unless ``stop_early`` is set to ``True``.
    """

    def __init__(
        self,
        *iterators: Iterable,
        stop_early: bool = False,
        weights: Optional[List[Union[int, float]]] = None,
        seed: int = 0,
    ) -> None:
        self.iterators = list(iterators)
        self.stop_early = stop_early
        self.seed = seed

        assert (
            len(self.iterators) > 1
        ), "There have to be at least two iterables to multiplex."

        if weights is None:
            self.weights = [1] * len(self.iterators)
        else:
            self.weights = weights

        assert len(self.iterators) == len(self.weights)

    def __iter__(self):
        rng = random.Random(self.seed)
        iters = [iter(it) for it in self.iterators]
        exhausted = [False for _ in range(len(iters))]

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
            selected = iters[idx]
            try:
                item = next(selected)
                yield item
            except StopIteration:
                exhausted[idx] = True
                continue

    def __len__(self) -> int:
        return sum(len(it) for it in self.iterators)

    def __add__(self, other) -> "LazyIteratorChain":
        return LazyIteratorChain(self, other)


class LazyShuffler(ImitatesDict):
    """
    A wrapper over an iterable that enables lazy shuffling.
    The shuffling algorithm is reservoir-sampling based.
    See :func:`lhotse.utils.streaming_shuffle` for details.
    """

    def __init__(
        self,
        iterator: Iterable,
        buffer_size: int = 10000,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.iterator = iterator
        self.buffer_size = buffer_size
        self.rng = rng

    def __iter__(self):
        return iter(
            streaming_shuffle(
                iter(self.iterator),
                bufsize=self.buffer_size,
                rng=self.rng,
            )
        )

    def __len__(self) -> int:
        return len(self.iterator)

    def __add__(self, other) -> "LazyIteratorChain":
        return LazyIteratorChain(self, other)


class LazyFilter(ImitatesDict):
    """
    A wrapper over an iterable that enables lazy filtering.
    It works like Python's `filter` built-in by applying the filter predicate
    to each element and yielding it further if predicate returned ``True``.
    """

    def __init__(self, iterator: Iterable, predicate: Callable[[Any], bool]) -> None:
        self.iterator = iterator
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
        return filter(self.predicate, self.iterator)

    def __add__(self, other) -> "LazyIteratorChain":
        return LazyIteratorChain(self, other)

    def __len__(self) -> int:
        raise NotImplementedError(
            "LazyFilter does not support __len__ because it would require "
            "iterating over the whole iterator, which is not possible in a lazy fashion. "
            "If you really need to know the length, convert to eager mode first using "
            "`.to_eager()`. Note that this will require loading the whole iterator into memory."
        )


class LazyMapper(ImitatesDict):
    """
    A wrapper over an iterable that enables lazy function evaluation on each item.
    It works like Python's `map` built-in by applying a callable ``fn``
    to each element ``x`` and yielding the result of ``fn(x)`` further.
    """

    def __init__(self, iterator: Iterable, fn: Callable[[Any], Any]) -> None:
        self.iterator = iterator
        self.fn = fn
        assert callable(self.fn), f"LazyMapper: 'fn' arg must be callable (got {fn})."
        if (
            isinstance(self.fn, types.LambdaType)
            and self.fn.__name__ == "<lambda>"
            and not is_module_available("dill")
        ):
            warnings.warn(
                "A lambda was passed to LazyMapper: it may prevent you from forking this process. "
                "If you experience issues with num_workers > 0 in torch.utils.data.DataLoader, "
                "try passing a regular function instead."
            )

    def __iter__(self):
        return map(self.fn, self.iterator)

    def __len__(self) -> int:
        return len(self.iterator)

    def __add__(self, other) -> "LazyIteratorChain":
        return LazyIteratorChain(self, other)


class LazyFlattener(ImitatesDict):
    """
    A wrapper over an iterable of collections that flattens it to an iterable of items.

    Example::

        >>> list_of_cut_sets: List[CutSet] = [CutSet(...), CutSet(...)]
        >>> list_of_cuts: List[Cut] = list(LazyFlattener(list_of_cut_sets))
    """

    def __init__(self, iterator: Iterable) -> None:
        self.iterator = iterator

    def __iter__(self):
        for cuts in self.iterator:
            yield from cuts

    def __add__(self, other) -> "LazyIteratorChain":
        return LazyIteratorChain(self, other)

    def __len__(self) -> int:
        raise NotImplementedError(
            "LazyFlattener does not support __len__ because it would require "
            "iterating over the whole iterator, which is not possible in a lazy fashion. "
            "If you really need to know the length, convert to eager mode first using "
            "`.to_eager()`. Note that this will require loading the whole iterator into memory."
        )


class LazyRepeater(ImitatesDict):
    """
    A wrapper over an iterable that enables to repeat it N times or infinitely (default).
    """

    def __init__(
        self, iterator: Iterable, times: Optional[int] = None, preserve_id: bool = False
    ) -> None:
        self.iterator = iterator
        self.times = times
        self.preserve_id = preserve_id
        assert self.times is None or self.times > 0

    def __iter__(self):
        epoch = 0
        while self.times is None or epoch < self.times:
            if self.preserve_id:
                iterator = self.iterator
            else:
                iterator = LazyMapper(
                    self.iterator, partial(attach_repeat_idx_to_id, idx=epoch)
                )

            yield from iterator
            epoch += 1

    def __len__(self) -> int:
        if self.times is None:
            raise AttributeError()
        return len(self.iterator) * self.times

    def __add__(self, other) -> "LazyIteratorChain":
        return LazyIteratorChain(self, other)


class LazySlicer(ImitatesDict):
    """
    A wrapper over an iterable that enables selecting k-th element every n elements.
    """

    def __init__(self, iterator: Iterable, k: int, n: int) -> None:
        self.iterator = iterator
        assert (
            k < n
        ), f"When selecting k-th element every n elements, k must be less than n (got k={k} n={n})."
        self.k = k
        self.n = n

    def __iter__(self):
        for idx, item in enumerate(self.iterator):
            if idx % self.n == self.k:
                yield item

    def __add__(self, other) -> "LazyIteratorChain":
        return LazyIteratorChain(self, other)

    def __len__(self) -> int:
        raise NotImplementedError(
            "LazySlicer does not support __len__ because it would require "
            "iterating over the whole iterator, which is not possible in a lazy fashion. "
            "If you really need to know the length, convert to eager mode first using "
            "`.to_eager()`. Note that this will require loading the whole iterator into memory."
        )


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
