import gzip
import itertools
import json
import warnings
from collections import deque
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, Optional, Type, Union

import numpy as np
import yaml

from lhotse.utils import Pathlike, ifnone, is_module_available

# TODO: figure out how to use some sort of typing stubs
#  so that linters/static checkers don't complain
Manifest = Any  # Union['RecordingSet', 'SupervisionSet', 'FeatureSet', 'CutSet']


def save_to_yaml(data: Any, path: Pathlike) -> None:
    compressed = str(path).endswith('.gz')
    opener = gzip.open if compressed else open
    mode = 'wt' if compressed else 'w'
    with opener(path, mode) as f:
        try:
            # When pyyaml is installed with C extensions, it can speed up the (de)serialization noticeably
            yaml.dump(data, stream=f, Dumper=yaml.CSafeDumper)
        except AttributeError:
            yaml.dump(data, stream=f, Dumper=yaml.SafeDumper)


def load_yaml(path: Pathlike) -> dict:
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path) as f:
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
    compressed = str(path).endswith('.gz')
    opener = gzip.open if compressed else open
    mode = 'wt' if compressed else 'w'
    with opener(path, mode) as f:
        json.dump(data, f, indent=2)


def load_json(path: Pathlike) -> Union[dict, list]:
    """Load a JSON file. Also supports compressed JSON with a ``.gz`` extension."""
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path) as f:
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
    compressed = str(path).endswith('.gz')
    opener = gzip.open if compressed else open
    mode = 'wt' if compressed else 'w'
    with opener(path, mode) as f:
        for item in data:
            print(json.dumps(item), file=f)


def load_jsonl(path: Pathlike) -> Generator[Dict[str, Any], None, None]:
    """Load a JSON file. Also supports compressed JSON with a ``.gz`` extension."""
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path) as f:
        for line in f:
            # The temporary variable helps fail fast
            ret = json.loads(line)
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
    writing them later. It can also be quried for existing IDs so that the user
    code may skip preparing the corresponding manifets.

    Example:

        >>> from lhotse import RecordingSet, Recording
        ... with RecordingSet.open_writer('recordings.jsonl.gz') as writer:
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

    def __init__(self, path: Pathlike, overwrite: bool = False) -> None:
        self.path = Path(path)
        assert extension_contains('.jsonl', self.path)
        self.compressed = extension_contains('.gz', self.path)
        self._open = gzip.open if self.compressed else open
        self.mode = 'wt' if self.compressed else 'w'
        self.ignore_ids = set()
        if self.path.is_file() and not overwrite:
            self.mode = 'at' if self.compressed else 'a'
            with self._open(self.path) as f:
                self.ignore_ids = {data['id'] for data in (json.loads(line) for line in f) if 'id' in data}

    def __enter__(self) -> 'SequentialJsonlWriter':
        self.file = self._open(self.path, self.mode)
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

    def write(self, manifest) -> None:
        """
        Serializes a manifest item (e.g. :class:`~lhotse.audio.Recording`,
        :class:`~lhotse.cut.Cut`, etc.) to JSON and stores it in a JSONL file.
        """
        try:
            if manifest.id in self.ignore_ids:
                return
        except AttributeError:
            pass
        print(
            json.dumps(manifest.to_dict(), cls=NumpyEncoder),
            file=self.file
        )


class JsonlMixin:
    def to_jsonl(self, path: Pathlike) -> None:
        save_to_jsonl(self.to_dicts(), path)

    @classmethod
    def from_jsonl(cls, path: Pathlike) -> Manifest:
        data = load_jsonl(path)
        return cls.from_dicts(data)

    @classmethod
    def open_writer(cls, path: Pathlike, overwrite: bool = False) -> SequentialJsonlWriter:
        """
        Open a sequential writer that allows to store the manifests one by one,
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
        writing them later. It can also be quried for existing IDs so that the user
        code may skip preparing the corresponding manifets.

        Example:

            >>> from lhotse import RecordingSet, Recording
            ... with RecordingSet.open_writer('recordings.jsonl.gz') as writer:
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
        return SequentialJsonlWriter(path, overwrite=overwrite)


class LazyMixin:
    @classmethod
    def from_jsonl_lazy(cls, path: Pathlike) -> Manifest:
        """
        Read a JSONL manifest in a lazy manner, using pyarrow.
        The contents of the file are loaded into memory,
        but in a memory-efficient format, and they are deserialized into
        Python objects such as Cut, Supervision etc. only upon request.

        In this mode, most operations on the manifest set may be very slow:
        including selecting specific manifests by their IDs, or splitting,
        shuffling, sorting, etc.
        However, iterating over the manifest is going to fairly fast.

        This method requires ``pyarrow`` and ``pandas`` to be installed.
        """
        return cls(LazyDict(path))

    def to_arrow(self, path: Pathlike) -> None:
        """
        Store the manifest in Apache Arrow streaming binary format.
        For very large manifests it can be ~5x larger that a corresponding compressed JSONL,
        but it allows to read the manifest with a relatively small memory footprint (~300M).
        """
        import pyarrow as pa
        # If the underlying storage for manifests is already lazy, we can
        # access the arrow tables directly without the need to convert items.
        if self.is_lazy:
            # TODO: I don't want to add a special method for retrieving those in each manifest type;
            #       after this work is done, I will make a refactoring PR that renames these members
            #       to sth like ".data" so that it's uniform across manifests.
            from lhotse import RecordingSet, SupervisionSet, CutSet
            if isinstance(self, RecordingSet):
                table = self.recordings.table
            elif isinstance(self, SupervisionSet):
                table = self.segments.table
            elif isinstance(self, CutSet):
                table = self.cuts.table
            else:
                raise NotImplementedError(f"Unsupported type of manifest for arrow serialization: {type(self)}")
            with open(path, "wb") as f, pa.RecordBatchFileWriter(f, schema=table.schema) as writer:
                for batch in table.to_batches():
                    writer.write_batch(batch)
        else:
            # We will take the first 1000 items from the manifest to infer the schema.
            # TODO: might want to sample items randomly in case their manifests vary...
            schema = pa.schema(pa.array(list(self.subset(first=1000).to_dicts())).type)
            # Open the file for writing and initialize the pyarrow batch writer.
            # Note that the batch size we determine here will be used to load whole chunks into
            # memory during deserialization.
            with open(path, "wb") as f, pa.RecordBatchFileWriter(f, schema=schema) as writer:
                # We are (lazily) grouping the items in manifest into chunks,
                # each of ``batch_size`` items.
                batch_size = 10 * 1024
                chunks = grouper(n=batch_size, iterable=self.to_dicts())
                for chunk in chunks:
                    # We convert the items in each chunk into Arrow's columnar representation.
                    # To do this, we first iterate by available "columns" (i.e. dict keys),
                    # and for each of them create an Arrow array with the corresponding values.
                    # These arrays are then used to create an arrow Table.
                    arrays = [
                        pa.array(
                            [item.get(key) for item in chunk],
                            type=schema.field(key_idx).type
                        ) for key_idx, key in enumerate(schema.names)
                    ]
                    table = pa.Table.from_arrays(arrays, schema=schema)
                    # The loop below will iterate only once, since we ensured there's exactly one batch.
                    for idx, batch in enumerate(table.to_batches(max_chunksize=batch_size)):
                        writer.write_batch(batch)

    @classmethod
    def from_arrow(cls, path: Pathlike) -> Manifest:
        """
        Read a manifest stored in Apache Arrow streaming binary format in a lazy manner.
        This method is supposed to use mmap, which should significantly ease
        the memory usage.
        The manifest items are deserialized into Python objects such as Cut,
        Supervision etc. only upon request.

        In this mode, most operations on the manifest set may be very slow:
        including selecting specific manifests by their IDs, or splitting,
        shuffling, sorting, etc.
        However, iterating over the manifest is going to fairly fast.

        This method requires ``pyarrow`` and ``pandas`` to be installed.
        """
        return cls(LazyDict(path))


def grouper(n, iterable):
    """https://stackoverflow.com/questions/8991506/iterate-an-iterator-by-chunks-of-n-in-python"""
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def extension_contains(ext: str, path: Path) -> bool:
    return any(ext == sfx for sfx in path.suffixes)


def load_manifest(path: Pathlike, manifest_cls: Optional[Type] = None) -> Manifest:
    """Generic utility for reading an arbitrary manifest."""
    from lhotse import CutSet, FeatureSet, RecordingSet, SupervisionSet
    # Determine the serialization format and read the raw data.
    path = Path(path)
    assert path.is_file(), f'No such path: {path}'
    if extension_contains('.jsonl', path):
        raw_data = load_jsonl(path)
        if manifest_cls is None:
            # Note: for now, we need to load the whole JSONL rather than read it in
            # a streaming way, because we have no way to know which type of manifest
            # we should decode later; since we're consuming the underlying generator
            # each time we try, not materializing the list first could lead to data loss
            raw_data = list(raw_data)
    elif extension_contains('.json', path):
        raw_data = load_json(path)
    elif extension_contains('.yaml', path):
        raw_data = load_yaml(path)
    elif extension_contains('.arrow', path):
        assert manifest_cls is not None, \
            "For lazy deserialization with arrow, the manifest type has to be known. " \
            "Try using [CutSet|RecordingSet|SupervisionSet].from_file(...) instead."
        return manifest_cls.from_arrow(path)
    else:
        raise ValueError(f"Not a valid manifest: {path}")
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
        raise ValueError(f'Unknown type of manifest: {path}')
    return data_set


def store_manifest(manifest: Manifest, path: Pathlike) -> None:
    path = Path(path)
    if extension_contains('.jsonl', path):
        manifest.to_jsonl(path)
    elif extension_contains('.json', path):
        manifest.to_json(path)
    elif extension_contains('.yaml', path):
        manifest.to_yaml(path)
    elif extension_contains('.arrow', path):
        manifest.to_arrow(path)
    else:
        raise ValueError(f"Unknown serialization format for: {path}")


class Serializable(JsonMixin, JsonlMixin, LazyMixin, YamlMixin):
    @classmethod
    def from_file(cls, path: Pathlike) -> Manifest:
        return load_manifest(path, manifest_cls=cls)

    def to_file(self, path: Pathlike) -> None:
        store_manifest(self, path)


def _check_arrow():
    if not is_module_available('pyarrow', 'pandas'):
        raise ImportError("In order to leverage lazy manifest capabilities of Lhotse, "
                          "please install additional, optional dependencies: "
                          "'pip install pyarrow pandas'")


class LazyDict:
    """
    LazyDict imitates a ``dict``, but it uses Apache Arrow (via pyarrow) to
    read the data on-the-fly from disk using mmap.

    This class is designed to be a "drop-in" replacement for ordinary dicts
    to support lazy loading of RecordingSet, SupervisionSet and CutSet.

    During initialization, Pyarrow scans a JSONL file using multithreaded
    native code and determines the JSON schema and the number of items.
    It is reasonably fast when iterated over, and quite slow when looking
    up single items (unless we are using it incorrectly, which is possible).
    Thanks to Pyarrow, we are able to open manifests with more than 10 million
    items in seconds and iterate over them with a small overhead.

    .. caution::
        This class is optimized for iteration or sequential access (i.e. iterating
        linearly over contiguous sequence of keys).
        Random access is possible but it may trigger a pessimistic complexity,
        making it incredibly slow...
    """

    def __init__(self, path: Pathlike):
        self.path = Path(path)
        self._reset()

    def __getstate__(self):
        """
        Store the state for pickling -- we'll only store the path, and re-initialize
        LazyDict when unpickled. This is necessary to transfer LazyDict across processes
        for PyTorch's DataLoader workers (otherwise mmapped file gets copied into memory).
        """
        state = {'path': self.path}
        return state

    def __setstate__(self, state):
        """Restore the state when unpickled -- open the mmap/jsonl file again."""
        self.__dict__.update(state)
        self._reset()

    def _reset(self):
        _check_arrow()
        self._init_table_from_path()
        self.batches = deque(self.table.to_batches())
        self.curr_view = self.batches[0].to_pandas()
        self.id2pos = dict(zip(self.curr_view.id, range(len(self.curr_view.id))))
        self.prev_view = None
        self.prev_id2pos = {}

    def _init_table_from_path(self):
        if '.jsonl' in self.path.suffixes:
            # Can read ".jsonl" or ".jsonl.gz"
            import pyarrow.json as paj
            self.table = paj.read_json(
                str(self.path),
                read_options=paj.ReadOptions(
                    # magic constants:
                    # 894 - estimated average number of bytes per JSON item manifest
                    # 10000 - how many items we want to have in a chunk (Arrow's "batch")
                    block_size=894 * 10000
                )
            )
        elif '.arrow' == self.path.suffixes[-1]:
            # Can read ".arrow"
            import pyarrow as pa
            mmap = pa.memory_map(str(self.path))
            stream = pa.ipc.open_file(mmap)
            self.table = stream.read_all()
        else:
            raise ValueError(f"Unknown LazyDict file format : '{self.path}'")

    def _progress(self):
        # Rotate the deque to the left by one item.
        # [0, 1, 2] -> [1, 2, 0]
        self.batches.rotate(-1)
        self.prev_view = self.curr_view
        self.curr_view = self.batches[0].to_pandas()
        self.prev_id2pos = self.id2pos
        self.id2pos = dict(zip(self.curr_view.id, range(len(self.curr_view.id))))

    def _find_key(self, key: str):
        # We will rotate the deque with N lazy views at most N times
        # to search for a given key.
        max_rotations = len(self.batches)
        for _ in range(max_rotations):
            # Try without any rotations in the first iteration --
            # this should make search faster for contiguous keys.
            pos = self.id2pos.get(key)
            if pos is not None:
                return deserialize_item(self.curr_view.iloc[pos].to_dict())
            pos = self.prev_id2pos.get(key)
            if pos is not None:
                return deserialize_item(self.prev_view.iloc[pos].to_dict())
            # Not found in the current Arrow's "batch" -- we'll advance
            # to the next one and try again.
            self._progress()
        # Key not found anyhwere.
        return None

    def __len__(self) -> int:
        return self.table.num_rows

    def __getitem__(self, key: str):
        """This is extremely inefficient and should not be used this way."""
        value = self._find_key(key)
        if value is None:
            raise KeyError(f"No such key: {key}")
        return value

    def get(self, key, or_=None):
        return ifnone(self._find_key(key), or_)

    def __contains__(self, key: str):
        value = self._find_key()
        return value is not None

    def __repr__(self):
        return f'LazyDict(num_items={len(self)})'

    def __iter__(self):
        for b in self.table.to_batches():
            yield from b['id'].tolist()

    def keys(self):
        return iter(self)

    def values(self):
        for b in self.table.to_batches():
            # This seems to be the fastest way to iterate rows in a pyarrow table.
            # Conversion to pandas seems to have the least overhead
            # due to Arrow's zero-copy memory sharing policy.
            yield from (deserialize_item(row.to_dict()) for idx, row in b.to_pandas().iterrows())

    def items(self):
        yield from ((cut.id, cut) for cut in self.values())


def deserialize_item(data: dict) -> Any:
    # Figures out what type of manifest is being decoded with some heuristics
    # and returns a Lhotse manifest object rather than a raw dict.
    from lhotse import MonoCut, Features, Recording, SupervisionSegment
    from lhotse.cut import MixedCut
    data = arr2list_recursive(data)
    if 'sources' in data:
        return Recording.from_dict(data)
    if 'num_features' in data:
        return Features.from_dict(data)
    if 'type' not in data:
        return SupervisionSegment.from_dict(data)
    cut_type = data.pop('type')
    if cut_type == 'MonoCut':
        return MonoCut.from_dict(data)
    if cut_type == 'Cut':
        warnings.warn('Your manifest was created with Lhotse version earlier than v0.8, when MonoCut was called Cut. '
                      'Please re-generate it with Lhotse v0.8 as it might stop working in a future version '
                      '(using manifest.from_file() and then manifest.to_file() should be sufficient).')
        return MonoCut.from_dict(data)
    if cut_type == 'MixedCut':
        return MixedCut.from_dict(data)
    raise ValueError(f"Unexpected cut type during deserialization: '{cut_type}'")


def arr2list_recursive(data: Union[dict, list], filter_none: bool = True) -> Union[dict, list]:
    """
    A helper method for converting dicts read via pyarrow,
    which have numpy arrays instead of scalars and regular lists.
    """
    return {
        # Array containing objects: go deeper
        k: [arr2list_recursive(x) for x in v] if isinstance(v, np.ndarray) and v.dtype == np.dtype('O')
        # Array (likely) containing numeric types: convert to list and to Python numeric types
        else v.tolist() if isinstance(v, (np.generic, np.ndarray))
        # Dict: go deeper
        else arr2list_recursive(v) if isinstance(v, dict)
        # Don't change anything
        else v
        for k, v in data.items()
        if v is not None or not filter_none
    }


class NumpyEncoder(json.JSONEncoder):
    """
    Utility that converts numpy types to native Python types for JSON serialization.

    Example:
        >>> with open('foo.json', 'w') as f:
        ...     json.dump({'a': np.arange(10)}, f, cls=NumpyEncoder)
    """
    def default(self, obj):
        if isinstance(obj, (np.generic, np.ndarray)):
            return obj.tolist()
        else:
            return super().default(obj)
