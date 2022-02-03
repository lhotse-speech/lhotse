import itertools
import json
import random
import warnings
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Type, Union

import yaml

from lhotse.utils import Pathlike, is_module_available
from lhotse.workarounds import gzip_open_robust

# TODO: figure out how to use some sort of typing stubs
#  so that linters/static checkers don't complain
Manifest = Any  # Union['RecordingSet', 'SupervisionSet', 'FeatureSet', 'CutSet']


def open_best(path: Pathlike, mode: str = "r"):
    if is_module_available("smart_open"):
        from smart_open import smart_open

        # This will work with JSONL anywhere that smart_open supports, e.g. cloud storage.
        open_fn = smart_open
    else:
        compressed = str(path).endswith(".gz")
        if compressed and "t" not in mode and "b" not in mode:
            # Opening as bytes not requested explicitly, use "t" to tell gzip to handle unicode.
            mode = mode + "t"
        open_fn = gzip_open_robust if compressed else open

    return open_fn(path, mode)


def save_to_yaml(data: Any, path: Pathlike) -> None:
    with open_best(path, "w") as f:
        try:
            # When pyyaml is installed with C extensions, it can speed up the (de)serialization noticeably
            yaml.dump(data, stream=f, Dumper=yaml.CSafeDumper)
        except AttributeError:
            yaml.dump(data, stream=f, Dumper=yaml.SafeDumper)


def load_yaml(path: Pathlike) -> dict:
    with open_best(path) as f:
        try:
            # When pyyaml is installed with C extensions, it can speed up the (de)serialization noticeably
            return yaml.load(stream=f, Loader=yaml.CSafeLoader)
        except AttributeError:
            return yaml.load(stream=f, Loader=yaml.SafeLoader)


class YamlMixin:
    def to_yaml(self, path: Pathlike) -> None:
        save_to_yaml(list(self.to_dicts()), path)

    @classmethod
    def from_yaml(cls, path: Pathlike) -> Manifest:
        data = load_yaml(path)
        return cls.from_dicts(data)


def save_to_json(data: Any, path: Pathlike) -> None:
    """Save the data to a JSON file. Will use GZip to compress it if the path ends with a ``.gz`` extension."""
    with open_best(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: Pathlike) -> Union[dict, list]:
    """Load a JSON file. Also supports compressed JSON with a ``.gz`` extension."""
    with open_best(path) as f:
        return json.load(f)


class JsonMixin:
    def to_json(self, path: Pathlike) -> None:
        save_to_json(list(self.to_dicts()), path)

    @classmethod
    def from_json(cls, path: Pathlike) -> Manifest:
        data = load_json(path)
        return cls.from_dicts(data)


def save_to_jsonl(data: Iterable[Dict[str, Any]], path: Pathlike) -> None:
    """Save the data to a JSON file. Will use GZip to compress it if the path ends with a ``.gz`` extension."""
    with open_best(path, "w") as f:
        for item in data:
            print(json.dumps(item), file=f)


def load_jsonl(path: Pathlike) -> Generator[Dict[str, Any], None, None]:
    """Load a JSON file. Also supports compressed JSON with a ``.gz`` extension."""
    with open_best(path) as f:
        for line in f:
            # The temporary variable helps fail fast
            ret = decode_json_line(line)
            yield ret


class SequentialJsonlWriter:
    """
    SequentialJsonlWriter allows to store the manifests one by one,
    without the necessity of storing the whole manifest set in-memory.
    Supports writing to JSONL format (``.jsonl``), with optional gzip
    compression (``.jsonl.gz``).

    Example:

        >>> from lhotse import RecordingSet
        ... recordings = [...]
        ... with RecordingSet.open_writer('recordings.jsonl.gz') as writer:
        ...     for recording in recordings:
        ...         writer.write(recording)

    This writer can be useful for continuing to write files that were previously
    stopped -- it will open the existing file and scan it for item IDs to skip
    writing them later. It can also be queried for existing IDs so that the user
    code may skip preparing the corresponding manifets.

    Example:

        >>> from lhotse import RecordingSet, Recording
        ... with RecordingSet.open_writer('recordings.jsonl.gz', overwrite=False) as writer:
        ...     for path in Path('.').rglob('*.wav'):
        ...         recording_id = path.stem
        ...         if writer.contains(recording_id):
        ...             # Item already written previously - skip processing.
        ...             continue
        ...         # Item doesn't exist yet - run extra work to prepare the manifest
        ...         # and store it.
        ...         recording = Recording.from_file(path, recording_id=recording_id)
        ...         writer.write(recording)
    """

    def __init__(self, path: Pathlike, overwrite: bool = True) -> None:
        self.path = path
        if not extension_contains(".jsonl", self.path):
            raise InvalidPathExtension(
                f"SequentialJsonlWriter supports only JSONL format (one JSON item per line), "
                f"but path='{path}'."
            )
        self.mode = "w"
        self.ignore_ids = set()
        if Path(self.path).is_file() and not overwrite:
            self.mode = "a"
            with open_best(self.path) as f:
                self.ignore_ids = {
                    data["id"]
                    for data in (decode_json_line(line) for line in f)
                    if "id" in data
                }

    def __enter__(self) -> "SequentialJsonlWriter":
        self.file = open_best(self.path, self.mode)
        return self

    def __exit__(self, *args, **kwargs) -> None:
        self.file.close()

    def __contains__(self, item: Union[str, Any]) -> bool:
        if isinstance(item, str):
            return item in self.ignore_ids
        try:
            return item.id in self.ignore_ids
        except AttributeError:
            # The only case when this happens is for the FeatureSet -- Features do not have IDs.
            # In that case we can't know if they are already written or not.
            return False

    def contains(self, item: Union[str, Any]) -> bool:
        return item in self

    def write(self, manifest: Any, flush: bool = False) -> None:
        """
        Serializes a manifest item (e.g. :class:`~lhotse.audio.Recording`,
        :class:`~lhotse.cut.Cut`, etc.) to JSON and stores it in a JSONL file.

        :param manifest: the manifest to be written.
        :param flush: should we flush the file after writing (ensures the changes
            are synced with the disk and not just buffered for later writing).
        """
        try:
            if manifest.id in self.ignore_ids:
                return
        except AttributeError:
            pass
        print(json.dumps(manifest.to_dict()), file=self.file)
        if flush:
            self.file.flush()

    def open_manifest(self) -> Manifest:
        """
        Opens the manifest that this writer has been writing to.
        The manifest is opened in a lazy mode.
        Returns ``None`` when the manifest is empty.
        """
        if not Path(self.path).exists():
            return None
        if not self.file.closed:
            # If the user hasn't finished writing, make sure the latest
            # changes are propagated.
            self.file.flush()
        return load_manifest_lazy(self.path)


class InvalidPathExtension(ValueError):
    pass


class InMemoryWriter:
    """
    Mimics :class:`.SequentialJsonlWriter` API but doesn't actually perform any I/O.
    It is used internally to create manifest sets without writing them to disk.
    """

    def __init__(self):
        self.items = []
        # for compatibility with SequentialJsonlWriter
        self.ignore_ids = frozenset()

    def __enter__(self) -> "InMemoryWriter":
        return self

    def __exit__(self, *args, **kwargs) -> None:
        pass

    def __contains__(self, item) -> bool:
        return False

    def contains(self, item: Union[str, Any]) -> bool:
        return item in self

    def write(self, manifest, flush: bool = False) -> None:
        self.items.append(manifest)

    def open_manifest(self) -> Optional[Manifest]:
        """
        Return a manifest set. Resolves the proper manifest set class by itself.
        Returns ``None`` when the manifest is empty.
        """
        if not self.items:
            return None
        cls = resolve_manifest_set_class(self.items[0])
        return cls.from_items(self.items)


class JsonlMixin:
    def to_jsonl(self, path: Pathlike) -> None:
        save_to_jsonl(self.to_dicts(), path)

    @classmethod
    def from_jsonl(cls, path: Pathlike) -> Manifest:
        data = load_jsonl(path)
        return cls.from_dicts(data)

    @classmethod
    def open_writer(
        cls, path: Union[Pathlike, None], overwrite: bool = True
    ) -> Union[SequentialJsonlWriter, InMemoryWriter]:
        """
        Open a sequential writer that allows to store the manifests one by one,
        without the necessity of storing the whole manifest set in-memory.
        Supports writing to JSONL format (``.jsonl``), with optional gzip
        compression (``.jsonl.gz``).

        .. note:: when ``path`` is ``None``, we will return a :class:`.InMemoryWriter`
            instead has the same API but stores the manifests in memory.
            It is convenient when you want to make disk saving optional.

        Example:

            >>> from lhotse import RecordingSet
            ... recordings = [...]
            ... with RecordingSet.open_writer('recordings.jsonl.gz') as writer:
            ...     for recording in recordings:
            ...         writer.write(recording)

        This writer can be useful for continuing to write files that were previously
        stopped -- it will open the existing file and scan it for item IDs to skip
        writing them later. It can also be queried for existing IDs so that the user
        code may skip preparing the corresponding manifets.

        Example:

            >>> from lhotse import RecordingSet, Recording
            ... with RecordingSet.open_writer('recordings.jsonl.gz', overwrite=False) as writer:
            ...     for path in Path('.').rglob('*.wav'):
            ...         recording_id = path.stem
            ...         if writer.contains(recording_id):
            ...             # Item already written previously - skip processing.
            ...             continue
            ...         # Item doesn't exist yet - run extra work to prepare the manifest
            ...         # and store it.
            ...         recording = Recording.from_file(path, recording_id=recording_id)
            ...         writer.write(recording)
        """
        if path is None:
            return InMemoryWriter()
        return SequentialJsonlWriter(path, overwrite=overwrite)


class LazyMixin:
    @classmethod
    def from_jsonl_lazy(cls, path: Pathlike) -> Manifest:
        """
        Read a JSONL manifest in a lazy manner, which opens the file but does not
        read it immediately. It is only suitable for sequential reads and iteration.

        .. warning:: Opening the manifest in this way might cause some methods that
            rely on random access to fail.
        """
        return cls(LazyJsonlIterator(path))


def grouper(n, iterable):
    """https://stackoverflow.com/questions/8991506/iterate-an-iterator-by-chunks-of-n-in-python"""
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def extension_contains(ext: str, path: Pathlike) -> bool:
    return any(ext == sfx for sfx in Path(path).suffixes)


def load_manifest(path: Pathlike, manifest_cls: Optional[Type] = None) -> Manifest:
    """Generic utility for reading an arbitrary manifest."""
    from lhotse import CutSet, FeatureSet, RecordingSet, SupervisionSet

    # Determine the serialization format and read the raw data.
    if extension_contains(".jsonl", path):
        raw_data = load_jsonl(path)
        if manifest_cls is None:
            # Note: for now, we need to load the whole JSONL rather than read it in
            # a streaming way, because we have no way to know which type of manifest
            # we should decode later; since we're consuming the underlying generator
            # each time we try, not materializing the list first could lead to data loss
            raw_data = list(raw_data)
    elif extension_contains(".json", path):
        raw_data = load_json(path)
    elif extension_contains(".yaml", path):
        raw_data = load_yaml(path)
    else:
        raise ValueError(f"Not a valid manifest (does the path exist?): {path}")
    data_set = None

    # The parse the raw data into Lhotse's data structures.
    # If the user provided a "type hint", use it; otherwise we will try to guess it.
    if manifest_cls is not None:
        candidates = [manifest_cls]
    else:
        candidates = [RecordingSet, SupervisionSet, FeatureSet, CutSet]
    for manifest_type in candidates:
        try:
            data_set = manifest_type.from_dicts(raw_data)
            break
        except Exception:
            pass
    if data_set is None:
        raise ValueError(f"Unknown type of manifest: {path}")
    return data_set


def load_manifest_lazy(path: Pathlike) -> Optional[Manifest]:
    """
    Generic utility for reading an arbitrary manifest from a JSONL file.
    Returns None when the manifest is empty.
    """
    assert extension_contains(".jsonl", path)
    raw_data = iter(load_jsonl(path))
    try:
        first = next(raw_data)
    except StopIteration:
        return None  # empty manifest
    item = deserialize_item(first)
    cls = resolve_manifest_set_class(item)
    return cls.from_jsonl_lazy(path)


def load_manifest_lazy_or_eager(path: Pathlike) -> Optional[Manifest]:
    """
    Generic utility for reading an arbitrary manifest.
    If possible, opens the manifest lazily, otherwise reads everything into memory.
    """
    if extension_contains(".jsonl", path):
        return load_manifest_lazy(path)
    else:
        return load_manifest(path)


def resolve_manifest_set_class(item):
    """Returns the right *Set class for a manifest, e.g. Recording -> RecordingSet."""
    from lhotse import (
        Features,
        FeatureSet,
        Recording,
        RecordingSet,
        SupervisionSegment,
        SupervisionSet,
    )
    from lhotse.cut import Cut, CutSet

    if isinstance(item, Recording):
        return RecordingSet
    if isinstance(item, SupervisionSegment):
        return SupervisionSet
    if isinstance(item, Cut):
        return CutSet
    if isinstance(item, Features):
        return FeatureSet
    raise ValueError(
        f"No corresponding 'Set' class is known for item of type: {type(item)}"
    )


def store_manifest(manifest: Manifest, path: Pathlike) -> None:
    if extension_contains(".jsonl", path):
        manifest.to_jsonl(path)
    elif extension_contains(".json", path):
        manifest.to_json(path)
    elif extension_contains(".yaml", path):
        manifest.to_yaml(path)
    else:
        raise ValueError(f"Unknown serialization format for: {path}")


class Serializable(JsonMixin, JsonlMixin, LazyMixin, YamlMixin):
    @classmethod
    def from_file(cls, path: Pathlike) -> Manifest:
        return load_manifest(path, manifest_cls=cls)

    def to_file(self, path: Pathlike) -> None:
        store_manifest(self, path)


class LazyJsonlIterator:
    """
    LazyJsonlIterator provides the ability to read Lhotse objects from a
    JSONL file on-the-fly, without reading its full contents into memory.

    This class is designed to be a partial "drop-in" replacement for ordinary dicts
    to support lazy loading of RecordingSet, SupervisionSet and CutSet.
    Since it does not support random access reads, some methods of these classes
    might not work properly.
    """

    def __init__(self, path: Pathlike) -> None:
        self.path = path
        assert extension_contains(".jsonl", self.path)

    def _reset(self) -> None:
        self._file = open_best(self.path)

    def __getstate__(self):
        """
        Store the state for pickling -- we'll only store the path, and re-initialize
        this iterator when unpickled. This is necessary to transfer this object across processes
        for PyTorch's DataLoader workers.
        """
        state = {"path": self.path}
        return state

    def __setstate__(self, state: Dict):
        """Restore the state when unpickled -- open the jsonl file again."""
        self.__dict__.update(state)

    def __iter__(self):
        self._reset()
        return self

    def __next__(self):
        line = next(self._file)
        data = decode_json_line(line)
        item = deserialize_item(data)
        return item

    def values(self):
        yield from self

    def keys(self):
        return (item.id for item in self)

    def items(self):
        return ((item.id, item) for item in self)

    def __len__(self) -> int:
        return count_newlines_fast(self.path)

    def __add__(self, other) -> "LazyIteratorChain":
        return LazyIteratorChain(self, other)


class LazyIteratorChain:
    """
    A thin wrapper over multiple iterators that enables to combine lazy manifests
    in Lhotse. It iterates all underlying iterables sequentially.
    """

    def __init__(self, *iterators: Iterable) -> None:
        self.iterators = []
        for it in iterators:
            # Auto-flatten LazyIteratorChain instances if any are passed
            if isinstance(it, LazyIteratorChain):
                for sub_it in it.iterators:
                    self.iterators.append(sub_it)
            else:
                self.iterators.append(it)

    def __iter__(self):
        return (item for it in self.iterators for item in it)

    def values(self):
        yield from self

    def keys(self):
        return (item.id for item in self)

    def items(self):
        return ((item.id, item) for item in self)

    def __len__(self) -> int:
        return sum(len(it) for it in self.iterators)

    def __add__(self, other) -> "LazyIteratorChain":
        return LazyIteratorChain(self, other)


class LazyIteratorMultiplexer:
    """
    A wrapper over multiple iterators that enables to combine lazy manifests in Lhotse.
    During iteration, unlike :class:`.LazyIteratorChain`, :class:`.LazyIteratorMultiplexer`
    at each step randomly selects the iterable used to yield an item.

    Since the iterables might be of different length, we provide a ``weights`` parameter
    to let the user decide which iterables should be sampled more frequently than others.
    When an iterable is exhausted, we will keep sampling from the other iterables, until
    we exhaust them all.
    """

    def __init__(
        self,
        *iterators: Iterable,
        weights: Optional[List[Union[int, float]]] = None,
        seed: int = 0
    ) -> None:
        self.iterators = list(iterators)
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
        while not all(exhausted):
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

    def values(self):
        yield from self

    def keys(self):
        return (item.id for item in self)

    def items(self):
        return ((item.id, item) for item in self)

    def __len__(self) -> int:
        return sum(len(it) for it in self.iterators)

    def __add__(self, other) -> "LazyIteratorChain":
        return LazyIteratorChain(self, other)


def deserialize_item(data: dict) -> Any:
    # Figures out what type of manifest is being decoded with some heuristics
    # and returns a Lhotse manifest object rather than a raw dict.
    from lhotse import MonoCut, Features, Recording, SupervisionSegment
    from lhotse.array import deserialize_array
    from lhotse.cut import MixedCut

    if "shape" in data or "array" in data:
        return deserialize_array(data)
    if "sources" in data:
        return Recording.from_dict(data)
    if "num_features" in data:
        return Features.from_dict(data)
    if "type" not in data:
        return SupervisionSegment.from_dict(data)
    cut_type = data.pop("type")
    if cut_type == "MonoCut":
        return MonoCut.from_dict(data)
    if cut_type == "Cut":
        warnings.warn(
            "Your manifest was created with Lhotse version earlier than v0.8, when MonoCut was called Cut. "
            "Please re-generate it with Lhotse v0.8 as it might stop working in a future version "
            "(using manifest.from_file() and then manifest.to_file() should be sufficient)."
        )
        return MonoCut.from_dict(data)
    if cut_type == "MixedCut":
        return MixedCut.from_dict(data)
    raise ValueError(f"Unexpected cut type during deserialization: '{cut_type}'")


def deserialize_custom_field(data: Optional[dict]) -> Optional[dict]:
    """
    Handles deserialization of manifests inside the custom field dictionary
    (e.g. from :class:`~lhotse.SupervisionSegment` or :class:`~lhotse.MonoCut`).

    Mutates the input in-place, and returns it.

    :param data: the contents of ``manifest.custom`` field.
    :return: ``custom`` field dict with deserialized manifests (if any),
        or None when input is None.
    """
    if data is None:
        return None

    from lhotse.array import deserialize_array

    # If any of the values in the input are also dicts,
    # it indicates that might be a serialized array manifest.
    # We'll try to deserialize it, and if there is an error,
    # we'll just leave it as it was.
    for key, value in data.items():
        if isinstance(value, dict):
            try:
                data[key] = deserialize_array(value)
            except:
                pass

    return data


def count_newlines_fast(path: Pathlike):
    """
    Counts newlines in a file using buffered chunk reads.
    The fastest possible option in Python according to:
    https://stackoverflow.com/a/68385697/5285891
    (This is a slightly modified variant of that answer.)
    """

    def _make_gen(reader):
        b = reader(2 ** 16)
        while b:
            yield b
            b = reader(2 ** 16)

    with open_best(path, "rb") as f:
        count = sum(buf.count(b"\n") for buf in _make_gen(f.read))
    return count


if is_module_available("orjson"):
    import orjson

    decode_json_line = orjson.loads
else:
    decode_json_line = json.loads
