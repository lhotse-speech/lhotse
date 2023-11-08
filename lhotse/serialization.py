import itertools
import json
import sys
import warnings
from codecs import StreamReader, StreamWriter
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, Optional, Type, Union

import yaml

from lhotse.utils import Pathlike, is_module_available
from lhotse.workarounds import gzip_open_robust

# TODO: figure out how to use some sort of typing stubs
#  so that linters/static checkers don't complain
Manifest = Any  # Union['RecordingSet', 'SupervisionSet', 'FeatureSet', 'CutSet']


def open_best(path: Pathlike, mode: str = "r"):
    """
    Auto-determine the best way to open the input path or URI.
    Uses ``smart_open`` when available to handle URLs and URIs.

    Supports providing "-" as input to read from stdin or save to stdout.

    If the input is prefixed with "pipe:", it will open a subprocess and redirect
    either stdin or stdout depending on the mode.
    The concept is similar to Kaldi's "generalized pipes", but uses WebDataset syntax.
    """
    if str(path) == "-":
        if mode == "r":
            return StdStreamWrapper(sys.stdin)
        elif mode == "w":
            return StdStreamWrapper(sys.stdout)
        else:
            raise ValueError(
                f"Cannot open stream for '-' with mode other 'r' or 'w' (got: '{mode}')"
            )

    if isinstance(path, (BytesIO, StringIO, StreamWriter, StreamReader)):
        return path

    if str(path).startswith("pipe:"):
        return open_pipe(path[5:], mode)

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


def open_pipe(cmd: str, mode: str):
    """
    Runs the command and redirects stdin/stdout depending on the mode.
    Returns a file-like object that can be read from or written to.
    """
    from lhotse.utils import Pipe

    return Pipe(cmd, mode=mode, shell=True, bufsize=8092)


def save_to_yaml(data: Any, path: Pathlike) -> None:
    with open_best(path, "w") as f:
        try:
            # When pyyaml is installed with C extensions, it can speed up the (de)serialization noticeably
            yaml.dump(data, stream=f, Dumper=yaml.CSafeDumper)
        except AttributeError:
            yaml.dump(data, stream=f, Dumper=yaml.SafeDumper)


def load_yaml(path: Pathlike) -> dict:
    with open_best(path, "r") as f:
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
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: Pathlike) -> Union[dict, list]:
    """Load a JSON file. Also supports compressed JSON with a ``.gz`` extension."""
    with open_best(path, "r") as f:
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
            print(json.dumps(item, ensure_ascii=False), file=f)


def load_jsonl(path: Pathlike) -> Generator[Dict[str, Any], None, None]:
    """Load a JSON file. Also supports compressed JSON with a ``.gz`` extension."""
    with open_best(path, "r") as f:
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
    code may skip preparing the corresponding manifests.

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
        self.file = None
        if not (extension_contains(".jsonl", self.path) or (self.path == "-")):
            raise InvalidPathExtension(
                f"SequentialJsonlWriter supports only JSONL format (one JSON item per line), "
                f"but path='{path}'."
            )
        self.mode = "w"
        self.ignore_ids = set()
        if Path(self.path).is_file() and not overwrite:
            self.mode = "a"
            with open_best(self.path, "r") as f:
                self.ignore_ids = {
                    data["id"]
                    for data in (decode_json_line(line) for line in f)
                    if "id" in data
                }

    def __enter__(self) -> "SequentialJsonlWriter":
        self._maybe_open()
        return self

    def __exit__(self, *args, **kwargs) -> None:
        self.close()

    def __contains__(self, item: Union[str, Any]) -> bool:
        if isinstance(item, str):
            return item in self.ignore_ids
        try:
            return item.id in self.ignore_ids
        except AttributeError:
            # The only case when this happens is for the FeatureSet -- Features do not have IDs.
            # In that case we can't know if they are already written or not.
            return False

    def _maybe_open(self):
        if self.file is None:
            self.file = open_best(self.path, self.mode)

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None

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
        self._maybe_open()
        if not isinstance(manifest, dict):
            manifest = manifest.to_dict()
        print(json.dumps(manifest, ensure_ascii=False), file=self.file)
        if flush:
            self.file.flush()

    def open_manifest(self) -> Optional[Manifest]:
        """
        Opens the manifest that this writer has been writing to.
        The manifest is opened in a lazy mode.
        Returns ``None`` when the manifest is empty.
        """
        if not Path(self.path).exists():
            return None
        if self.file is not None and not self.file.closed:
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
        code may skip preparing the corresponding manifests.

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
    def from_items(self, data: Iterable):
        """
        Function to be implemented by every sub-class of this mixin.
        It's expected to create a sub-class instance out of an iterable of items
        that are held by the sub-class (e.g., ``CutSet.from_items(iterable_of_cuts)``).
        """
        raise NotImplemented

    @property
    def data(self) -> Union[Dict[str, Any], Iterable[Any]]:
        """
        Property to be implemented by every sub-class of this mixin.
        It can either be a regular Python dict holding manifests for eager mode,
        or a special iterator class for lazy mode.
        """
        raise NotImplemented

    def to_eager(self):
        """
        Evaluates all lazy operations on this manifest, if any, and returns a copy
        that keeps all items in memory.
        If the manifest was "eager" already, this is a no-op and won't copy anything.
        """
        if not self.is_lazy:
            return self
        cls = type(self)
        return cls.from_items(self)

    @property
    def is_lazy(self) -> bool:
        """
        Indicates whether this manifest was opened in lazy (read-on-the-fly) mode or not.
        """
        return not isinstance(self.data, (dict, list, tuple))

    @classmethod
    def from_jsonl_lazy(cls, path: Pathlike) -> Manifest:
        """
        Read a JSONL manifest in a lazy manner, which opens the file but does not
        read it immediately. It is only suitable for sequential reads and iteration.

        .. warning:: Opening the manifest in this way might cause some methods that
            rely on random access to fail.
        """
        from lhotse.lazy import LazyManifestIterator

        return cls(LazyManifestIterator(path))


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
            if len(data_set) == 0:
                raise RuntimeError()
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
    assert extension_contains(".jsonl", path) or str(path) == "-"
    raw_data = iter(load_jsonl(path))
    try:
        first = next(raw_data)
    except StopIteration:
        return None  # empty manifest
    item = deserialize_item(first)
    cls = resolve_manifest_set_class(item)
    return cls.from_jsonl_lazy(path)


def load_manifest_lazy_or_eager(
    path: Pathlike, manifest_cls=None
) -> Optional[Manifest]:
    """
    Generic utility for reading an arbitrary manifest.
    If possible, opens the manifest lazily, otherwise reads everything into memory.
    """
    if extension_contains(".jsonl", path) or str(path) == "-":
        return load_manifest_lazy(path)
    else:
        return load_manifest(path, manifest_cls=manifest_cls)


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
    raise NotALhotseManifest(
        f"No corresponding 'Set' class is known for item of type: {type(item)}"
    )


class NotALhotseManifest(Exception):
    pass


def store_manifest(manifest: Manifest, path: Pathlike) -> None:
    if extension_contains(".jsonl", path) or str(path) == "-":
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
        return load_manifest_lazy_or_eager(path, manifest_cls=cls)

    def to_file(self, path: Pathlike) -> None:
        store_manifest(self, path)


def deserialize_item(data: dict) -> Any:
    # Figures out what type of manifest is being decoded with some heuristics
    # and returns a Lhotse manifest object rather than a raw dict.
    from lhotse import Features, MonoCut, MultiCut, Recording, SupervisionSegment
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
    if cut_type == "MultiCut":
        return MultiCut.from_dict(data)
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
    from lhotse.audio import Recording

    # If any of the values in the input are also dicts,
    # it indicates that might be a serialized array manifest.
    # We'll try to deserialize it, and if there is an error,
    # we'll just leave it as it was.
    for key, value in data.items():
        if isinstance(value, dict):
            if all(k in value for k in ("id", "sources", "sampling_rate")):
                data[key] = Recording.from_dict(value)
                continue
            try:
                data[key] = deserialize_array(value)
            except:
                pass

    return data


if is_module_available("orjson"):
    import orjson

    def decode_json_line(line):
        try:
            return orjson.loads(line)
        except:
            return json.loads(line)

else:
    decode_json_line = json.loads


class StdStreamWrapper:
    def __init__(self, stream):
        self.stream = stream

    def close(self):
        pass

    def __enter__(self):
        return self.stream

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __getattr__(self, item: str):
        if item == "close":
            return self.close
        return getattr(self.stream, item)
