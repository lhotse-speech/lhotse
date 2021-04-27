import gzip
import json
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


class JsonlMixin:
    def to_jsonl(self, path: Pathlike) -> None:
        save_to_jsonl(self.to_dicts(), path)

    @classmethod
    def from_jsonl(cls, path: Pathlike) -> Manifest:
        data = load_jsonl(path)
        return cls.from_dicts(data)

    @classmethod
    def from_jsonl_lazy(cls, path: Pathlike) -> Manifest:
        """
        Read a manifest in a lazy manner, using pyarrow and mmap.
        The contents of the file are not loaded to memory immediately --
        we will only load them once they are requested.

        In this mode, most operations on the manifest set may be very slow:
        including selecting specific manifests by their IDs, or splitting,
        shuffling, sorting, etc.
        However, iterating over the manifest is going to fairly fast.

        This method requires ``pyarrow`` and ``pandas`` to be installed.
        """
        return cls(LazyDict(path))


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
    else:
        raise ValueError(f"Unknown serialization format for: {path}")


class Serializable(JsonMixin, JsonlMixin, YamlMixin):
    @classmethod
    def from_file(cls, path: Pathlike) -> Manifest:
        return load_manifest(path, manifest_cls=cls)

    def to_file(self, path: Pathlike) -> None:
        store_manifest(self, path)


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

    .. caution:
        We discourage using this like ``'cut-id' in lazy_dict`` or
        ``lazy_dict['cut-id']`` -- it is going to be much slower than iteration,
        which pre-loads chunks of the manifests.
    """

    def __init__(self, path: Pathlike):
        if not is_module_available('pyarrow', 'pandas'):
            raise ImportError("In order to leverage lazy manifest capabilities of Lhotse, "
                              "please install additional, optional dependencies: "
                              "'pip install pyarrow pandas'")
        import pyarrow.json as paj
        self.table = paj.read_json(str(path))
        self.batches = deque(self.table.to_batches())
        self.curr_view = self.batches[0].to_pandas()

    def _progress(self):
        self.batches.rotate()
        self.curr_view = self.batches[0].to_pandas()

    def _find_key(self, key: str):
        # We will rotate the deque with N lazy views at most N times
        # to search for a given key.
        max_rotations = len(self.batches)
        for _ in range(max_rotations):
            # Try without any rotations in the first iteration --
            # this should make search faster for contiguous keys.
            match = self.curr_view.query(f'id == "{key}"')
            if len(match):
                return self._deserialize_one(match.iloc[0].to_dict())
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
        return f'LazyDict(num_items={self.table.num_rows})'

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
            yield from (self._deserialize_one(row.to_dict()) for idx, row in b.to_pandas().iterrows())

    def items(self):
        yield from ((cut.id, cut) for cut in self.values())

    @staticmethod
    def _deserialize_one(data: dict) -> Any:
        # Figures out what type of manifest is being decoded with some heuristics
        # and returns a Lhotse manifest object rather than a raw dict.
        from lhotse import Cut, Features, Recording, SupervisionSegment
        from lhotse.cut import MixedCut
        data = arr2list_recursive(data)
        if 'sources' in data:
            return Recording.from_dict(data)
        if 'num_features' in data:
            return Features.from_dict(data)
        if 'type' not in data:
            return SupervisionSegment.from_dict(data)
        cut_type = data.pop('type')
        if cut_type == 'Cut':
            return Cut.from_dict(data)
        if cut_type == 'MixedCut':
            return MixedCut.from_dict(data)
        raise ValueError(f"Unexpected cut type during deserialization: '{cut_type}'")


def arr2list_recursive(data: Union[dict, list]) -> Union[dict, list]:
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
