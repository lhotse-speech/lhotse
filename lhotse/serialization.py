import gzip
import json
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, Union

import yaml

from lhotse.utils import Pathlike


def save_to_yaml(data: Any, path: Pathlike):
    compressed = str(path).endswith('.gz')
    opener = gzip.open if compressed else open
    mode = 'wt' if compressed else 'w'
    with opener(path, mode) as f:
        try:
            # When pyyaml is installed with C extensions, it can speed up the (de)serialization noticeably
            return yaml.dump(data, stream=f, Dumper=yaml.CSafeDumper)
        except AttributeError:
            return yaml.dump(data, stream=f, Dumper=yaml.SafeDumper)


def load_yaml(path: Pathlike) -> dict:
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path) as f:
        try:
            # When pyyaml is installed with C extensions, it can speed up the (de)serialization noticeably
            return yaml.load(stream=f, Loader=yaml.CSafeLoader)
        except AttributeError:
            return yaml.load(stream=f, Loader=yaml.SafeLoader)


class YamlMixin:
    def to_yaml(self, path: Pathlike):
        save_to_yaml(self.to_dicts(), path)

    @classmethod
    def from_yaml(cls, path: Pathlike):
        data = load_yaml(path)
        return cls.from_dicts(data)


def save_to_json(data: Any, path: Pathlike):
    """Save the data to a JSON file. Will use GZip to compress it if the path ends with a ``.gz`` extension."""
    compressed = str(path).endswith('.gz')
    opener = gzip.open if compressed else open
    mode = 'wt' if compressed else 'w'
    with opener(path, mode) as f:
        return json.dump(data, f, indent=2)


def load_json(path: Pathlike) -> Union[dict, list]:
    """Load a JSON file. Also supports compressed JSON with a ``.gz`` extension."""
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path) as f:
        return json.load(f)


class JsonMixin:
    def to_json(self, path: Pathlike):
        save_to_json(self.to_dicts(), path)

    @classmethod
    def from_json(cls, path: Pathlike):
        data = load_json(path)
        return cls.from_dicts(data)


def save_to_jsonl(data: Iterable[Dict[str, Any]], path: Pathlike):
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
    def to_jsonl(self, path: Pathlike):
        save_to_jsonl(self.to_dicts(), path)

    @classmethod
    def from_jsonl(cls, path: Pathlike):
        data = load_jsonl(path)
        return cls.from_dicts(data)


def extension_contains(ext: str, path: Path) -> bool:
    return any(ext == sfx for sfx in path.suffixes)


def load_manifest(path: Pathlike):
    """Generic utility for reading an arbitrary manifest."""
    from lhotse import CutSet, FeatureSet, RecordingSet, SupervisionSet
    path = Path(path)
    assert path.is_file(), f'No such path: {path}'
    if extension_contains('.jsonl', path):
        raw_data = load_jsonl(path)
    elif extension_contains('.json', path):
        raw_data = load_json(path)
    elif extension_contains('.yaml', path):
        raw_data = load_yaml(path)
    else:
        raise ValueError(f"Not a valid manifest: {path}")
    data_set = None
    for manifest_type in [RecordingSet, SupervisionSet, FeatureSet, CutSet]:
        try:
            data_set = manifest_type.from_dicts(raw_data)
        except Exception:
            pass
    if data_set is None:
        raise ValueError(f'Unknown type of manifest: {path}')
    return data_set


class Serializable(JsonMixin, JsonlMixin, YamlMixin):
    pass
