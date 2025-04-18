import itertools
import json
import os
import sys
import warnings
from codecs import StreamReader, StreamWriter
from contextlib import contextmanager
from functools import lru_cache
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Type, Union

import yaml
from packaging.version import parse as parse_version

from lhotse.utils import (
    Pathlike,
    Pipe,
    SmartOpen,
    is_module_available,
    is_valid_url,
    replace_bucket_with_profile_name,
)
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
    if isinstance(path, (BytesIO, StringIO, StreamWriter, StreamReader)):
        return path
    assert isinstance(
        path, (str, Path)
    ), f"Unexpected identifier type {type(path)} for object {path}. Expected str or pathlib.Path."
    try:
        return get_current_io_backend().open(path, mode)
    except Exception:
        if is_valid_url(path):
            raise ValueError(
                f"Error trying to open what seems to be a URI: '{path}'\n"
                f"In order to open URLs/URIs please run 'pip install smart_open' "
                f"(if you're trying to use AIStore, either the Python SDK is not installed (pip install aistore) "
                f"or {AIS_ENDPOINT_ENVVAR} is not defined."
            )
        raise


AIS_ENDPOINT_ENVVAR = "AIS_ENDPOINT"


@lru_cache
def is_aistore_available() -> bool:
    return AIS_ENDPOINT_ENVVAR in os.environ and is_valid_url(
        os.environ[AIS_ENDPOINT_ENVVAR]
    )


@lru_cache
def get_aistore_client():
    if not is_module_available("aistore"):
        raise ImportError(
            "Please run 'pip install aistore' in order to read data from AIStore."
        )
    if not is_aistore_available():
        raise ValueError(
            "Set a valid URL as AIS_ENDPOINT environment variable's value to read data from AIStore."
        )
    import aistore

    endpoint_url = os.environ["AIS_ENDPOINT"]
    version = parse_version(aistore.__version__)
    return aistore.Client(endpoint_url, timeout=(1, 20)), version


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


class IOBackend:
    """
    Base class for IO backends supported by Lhotse.
    An IO backend supports open() operations for reads and/or writes to file-like objects.
    Deriving classes are auto-registered under their class name, and auto-discoverable
    through functions:

    * :func:`~lhotse.serialization.available_io_backends`

    * :func:`~lhotse.serialization.get_current_io_backend`

    * :func:`~lhotse.serialization.set_current_io_backend`

    The default composite backend that tries to figure out the best solution
    can be obtained via :func:`~lhotse.serialization.get_default_io_backend`.

    New IO backends are expected to define the following methods:

    * `open(identifier: str, mode: str)` which returns a file-like object.
        Must be implemented.

    * `is_applicable(identifier: str) -> bool` returns `True` if a given
        backend can be used for a given identifier. True by default.

    * `is_available(identifier: str) -> bool` Class method. Only define it when
        the availability of the backend depends on some special actions,
        such as installing an option dependency.

    * `handles_special_case(identifier: str) -> bool` defined ONLY when
        a given IO Backend MUST be selected for a specific identifier.
        For example, only :class:`~lhotse.serialization.PipeIOBackend` handles
        piped commands like `"pipe:gunzip -c manifest.jsonl.gz"`.
    """

    KNOWN_BACKENDS = {}

    def __init_subclass__(cls, **kwargs):
        if cls.__name__ not in IOBackend.KNOWN_BACKENDS:
            IOBackend.KNOWN_BACKENDS[cls.__name__] = cls
        super().__init_subclass__(**kwargs)

    @classmethod
    def new(cls, name: str) -> "IOBackend":
        if name not in cls.KNOWN_BACKENDS:
            raise RuntimeError(f"Unknown IO backend name: {name}")
        return cls.KNOWN_BACKENDS[name]()

    def open(self, identifier: Pathlike, mode: str):
        raise NotImplementedError()

    @classmethod
    def is_available(cls) -> bool:
        return True

    def handles_special_case(self, identifier: Pathlike) -> bool:
        return False

    def is_applicable(self, identifier: Pathlike) -> bool:
        return True


class BuiltinIOBackend(IOBackend):
    """Calls Python's built-in `open`."""

    def open(self, identifier: Pathlike, mode: str):
        return open(identifier, mode=mode)

    def is_applicable(self, identifier: Pathlike) -> bool:
        return not is_valid_url(identifier)


class RedirectIOBackend(IOBackend):
    """Opens a stream to stdin or stdout."""

    def open(self, identifier: Pathlike, mode: str):
        if mode == "r":
            return StdStreamWrapper(sys.stdin)
        elif mode == "w":
            return StdStreamWrapper(sys.stdout)
        raise ValueError(
            f"Cannot open stream for '-' with mode other 'r' or 'w' (got: '{mode}')"
        )

    def handles_special_case(self, identifier: Pathlike) -> bool:
        return identifier == "-"

    def is_applicable(self, identifier: Pathlike) -> bool:
        return self.handles_special_case(identifier)


class PipeIOBackend(IOBackend):
    """Executes the provided command / pipe and wraps it into a file-like object."""

    def open(self, identifier: Pathlike, mode: str):
        """
        Runs the command and redirects stdin/stdout depending on the mode.
        Returns a file-like object that can be read from or written to.
        """
        return Pipe(str(identifier)[5:], mode=mode, shell=True, bufsize=8092)

    def handles_special_case(self, identifier: Pathlike) -> bool:
        return str(identifier).startswith("pipe:")

    def is_applicable(self, identifier: Pathlike) -> bool:
        return self.handles_special_case(identifier)


class GzipIOBackend(IOBackend):
    """Uses gzip.open to automatically (de)compress."""

    def open(self, identifier: Pathlike, mode: str):
        if "t" not in mode and "b" not in mode:
            # Opening as bytes not requested explicitly, use "t" to tell gzip to handle unicode.
            mode = mode + "t"
        return gzip_open_robust(identifier, mode)

    def handles_special_case(self, identifier: Pathlike) -> bool:
        identifier = str(identifier)
        return identifier.endswith(".gz") and not is_valid_url(identifier)

    def is_applicable(self, identifier: Pathlike) -> bool:
        return self.handles_special_case(identifier)


class SmartOpenIOBackend(IOBackend):
    """Uses `smart_open` library (if installed) to auto-determine how to handle the URI."""

    def open(self, identifier: Pathlike, mode: str):
        return SmartOpen.open(identifier, mode)

    @classmethod
    def is_available(cls) -> bool:
        return is_module_available("smart_open")


class AIStoreIOBackend(IOBackend):
    """
    Uses `aistore` client (if installed and enabled via AIS_ENDPOINT env var)
    to download data from AIStore if the identifier is a URL/URI.
    """

    def open(self, identifier: str, mode: str):
        client, version = get_aistore_client()
        object = client.fetch_object_by_url(identifier)
        if "r" in mode:
            try:
                # AIStore >= 1.10.0
                request = object.get_reader()
            except AttributeError:
                # AIStore < 1.10.0 deprecated method
                request = object.get()
            if version >= parse_version("1.9.1"):
                # AIStore SDK 1.9.1 supports ObjectFile for improved read fault resiliency
                return request.as_file()
            else:
                return request.raw()
        if "w" in mode:
            assert version >= parse_version("1.10.0"), (
                f"Writing to AIStore requires at least version 1.10.0 of AIStore Python SDK, "
                f"but your version is {version}"
            )
            return object.get_writer().as_file()

    @classmethod
    def is_available(cls) -> bool:
        return (
            is_module_available("aistore")
            and AIS_ENDPOINT_ENVVAR in os.environ
            and is_valid_url(os.environ[AIS_ENDPOINT_ENVVAR])
        )

    def handles_special_case(self, identifier: Pathlike) -> bool:
        return str(identifier).startswith("ais://")

    def is_applicable(self, identifier: Pathlike) -> bool:
        return is_valid_url(identifier)


def get_lhotse_msc_override_protocols() -> Any:
    return os.getenv("LHOTSE_MSC_OVERRIDE_PROTOCOLS", None)


def get_lhotse_msc_profile() -> Any:
    return os.getenv("LHOTSE_MSC_PROFILE", None)


def get_lhotse_msc_backend_forced() -> Any:
    """
    If set to True, the MSC backend will be forced to be used for regular URLs.
    """
    val = os.getenv("LHOTSE_MSC_BACKEND_FORCED", "False")
    return val.lower() == "true"


MSC_PREFIX = "msc"


class MSCIOBackend(IOBackend):
    """
    Uses Multi-Storage Client to download data from object store.

    Multi-Storage Client (MSC) is a Python library aims at providing a unified interface to object and file
    storage backends, including S3, GCS, AIStore, and more.  With no code change, user can seamlessly switch
    between different storage backends with corresponding MSC urls.

    To use MSCIOBackend, user will need

    1)
    MSC config file that specifies the storage backend information. Please refer to the MSC documentation
    for more details: https://nvidia.github.io/multi-storage-client/user_guide/quickstart.html#configuration

    2)
    Provide MSC URLs, OR
    Set env `LHOTSE_MSC_BACKEND_FORCED` to True to force the use of MSC backend for regular URLs.

    To learn more about MSC, please check out the GitHub repo: https://github.com/NVIDIA/multi-storage-client
    """

    def open(self, identifier: str, mode: str):
        """
        Convert identifier if is not prefixed with msc, and use msc.open to access the file
        For paths that are prefixed with msc, e.g. msc://profile/path/to/my/object1

        For paths are yet to migrate to msc-compatible url, e.g. protocol://bucket/path/to/my/object2
        1. override protocols provided by env LHOTSE_MSC_OVERRIDE_PROTOCOLS to msc: msc://bucket/path/to/my/object2
        2. override the profile/bucket name by env LHOTSE_MSC_PROFILE if provided: msc://profile/path/to/my/object2,
        if bucket name is not provided, then we expect the msc profile name to match with bucket name
        """
        if not is_module_available("multistorageclient"):
            raise RuntimeError(
                "Please run 'pip install multistorageclient' in order to use MSCIOBackend."
            )

        import multistorageclient as msc

        # if url prefixed with msc, then return early
        if MSCIOBackend.is_msc_url(identifier):
            return msc.open(identifier, mode)

        # override protocol if provided
        lhotse_msc_override_protocols = get_lhotse_msc_override_protocols()
        if lhotse_msc_override_protocols:
            if "," in lhotse_msc_override_protocols:
                override_protocol_list = lhotse_msc_override_protocols.split(",")
            else:
                override_protocol_list = [lhotse_msc_override_protocols]
            for override_protocol in override_protocol_list:
                if identifier.startswith(override_protocol):
                    identifier = identifier.replace(override_protocol, MSC_PREFIX)
                    break

        # override bucket if provided
        lhotse_msc_profile = get_lhotse_msc_profile()
        if lhotse_msc_profile:
            identifier = replace_bucket_with_profile_name(
                identifier, lhotse_msc_profile
            )

        try:
            file = msc.open(identifier, mode)
        except Exception as e:
            print(f"exception: {e}, identifier: {identifier}")
            raise e

        return file

    @classmethod
    def is_available(cls) -> bool:
        return is_module_available("multistorageclient")

    def handles_special_case(self, identifier: Pathlike) -> bool:
        return MSCIOBackend.is_msc_url(identifier)

    def is_applicable(self, identifier: Pathlike) -> bool:
        return is_module_available("multistorageclient") and (
            MSCIOBackend.is_msc_url(identifier)
            or (get_lhotse_msc_backend_forced() and is_valid_url(identifier))
        )

    @staticmethod
    def is_msc_url(identifier: Any) -> bool:
        return str(identifier).startswith(f"{MSC_PREFIX}://")


class CompositeIOBackend(IOBackend):
    """
    Composes multiple IO backends together.
    Uses `handles_special_case` and `is_applicable` of sub-backends to auto-detect
    which backend to select.

    In case of `handles_special_case`, if multiple backends could have worked,
    we'll use the first one in the list.
    """

    def __init__(self, backends: List[IOBackend]):
        self.backends = backends

    def open(self, identifier: Pathlike, mode: str):
        for b in self.backends:
            if b.handles_special_case(identifier):
                return b.open(identifier, mode)

        for b in self.backends:
            if b.is_applicable(identifier):
                return b.open(identifier, mode)

        raise RuntimeError(
            f"Couldn't find a suitable IOBackend for input '{identifier}'"
        )

    def handles_special_case(self, identifier: Pathlike) -> bool:
        return any(b.handles_special_case(identifier) for b in self.backends)

    def is_applicable(self, identifier: Pathlike) -> bool:
        return any(b.is_applicable(identifier) for b in self.backends)


CURRENT_IO_BACKEND: Optional["IOBackend"] = None


def available_io_backends() -> List[str]:
    """
    Return a list of names of available IO backends, including "default".
    """
    return ["default"] + sorted(
        b
        for b in IOBackend.KNOWN_BACKENDS
        if IOBackend.KNOWN_BACKENDS[b].is_available()
    )


@contextmanager
def io_backend(backend: Union["IOBackend", str]) -> Generator["IOBackend", None, None]:
    """
    Context manager that sets Lhotse's IO backend to the specified value
    and restores the previous IO backend at the end of its scope.

    Example::

        >>> with io_backend("AIStoreIOBackend"):
        ...     cuts = CutSet.from_file(...)  # forced open() via AIStore client
    """
    previous = get_current_io_backend()
    b = set_current_io_backend(backend)
    yield b
    set_current_io_backend(previous)


def get_current_io_backend() -> "IOBackend":
    """
    Return the backend currently set by the user, or default.
    """
    global CURRENT_IO_BACKEND

    # First check if the user has programmatically overridden the backend.
    if CURRENT_IO_BACKEND is not None:
        return CURRENT_IO_BACKEND

    # Then, check if the user has overridden the audio backend via an env var.
    maybe_backend = os.environ.get("LHOTSE_IO_BACKEND")
    if maybe_backend is not None:
        return set_current_io_backend(maybe_backend)

    # Lastly, fall back to the default backend.
    return set_current_io_backend("default")


def set_current_io_backend(backend: Union["IOBackend", str]) -> "IOBackend":
    """
    Force Lhotse to use a specific IO backend to open every path/URL/URI,
    overriding the default behaviour of "educated guessing".

    Example forcing Lhotse to use ``aistore`` library for every IO open() operation::

        >>> set_current_io_backend(AIStoreIOBackend())
    """
    global CURRENT_IO_BACKEND
    if backend == "default":
        backend = get_default_io_backend()
    elif isinstance(backend, str):
        backend = IOBackend.new(backend)
    else:
        if isinstance(backend, type):
            backend = backend()
        assert isinstance(
            backend, IOBackend
        ), f"Expected str or IOBackend, got: {backend}"
    CURRENT_IO_BACKEND = backend
    return CURRENT_IO_BACKEND


@lru_cache(maxsize=1)
def get_default_io_backend() -> "IOBackend":
    """
    Return a composite backend that auto-infers which internal backend can support reading
    from a given identifier.

    It first looks for special cases that need very specific handling
    (such as: stdin/stdout redirects, pipes)
    and tries to match them against relevant IO backends.
    """
    # Start with the special cases.
    backends = [
        RedirectIOBackend(),
        PipeIOBackend(),
    ]
    if MSCIOBackend.is_available():
        backends.append(MSCIOBackend())
    if AIStoreIOBackend.is_available():
        # Try AIStore before other generalist backends,
        # but only if it's installed and enabled via AIS_ENDPOINT env var.
        backends.append(AIStoreIOBackend())
    if SmartOpenIOBackend.is_available():
        backends.append(SmartOpenIOBackend())
    backends += [
        GzipIOBackend(),
        BuiltinIOBackend(),
    ]
    return CompositeIOBackend(backends)
