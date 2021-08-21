import gzip
import itertools
import json
import warnings
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, Optional, Type, Union

import yaml

from lhotse.utils import Pathlike

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
            json.dumps(manifest.to_dict()),
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
    def open_writer(cls, path: Pathlike, overwrite: bool = True) -> SequentialJsonlWriter:
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
        self.path = Path(path)
        assert extension_contains('.jsonl', self.path)

    def _reset(self) -> None:
        opener = gzip.open if str(self.path).endswith('.gz') else open
        self._file = opener(self.path)

    def __getstate__(self):
        """
        Store the state for pickling -- we'll only store the path, and re-initialize
        this iterator when unpickled. This is necessary to transfer this object across processes
        for PyTorch's DataLoader workers.
        """
        state = {'path': self.path}
        return state

    def __setstate__(self, state: Dict):
        """Restore the state when unpickled -- open the jsonl file again."""
        self.__dict__.update(state)

    def __iter__(self):
        self._reset()
        return self

    def __next__(self):
        line = next(self._file)
        data = json.loads(line)
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


def deserialize_item(data: dict) -> Any:
    # Figures out what type of manifest is being decoded with some heuristics
    # and returns a Lhotse manifest object rather than a raw dict.
    from lhotse import MonoCut, Features, Recording, SupervisionSegment
    from lhotse.cut import MixedCut
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

    path = Path(path)
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, "rb") as f:
        count = sum(buf.count(b"\n") for buf in _make_gen(f.read))
    return count
