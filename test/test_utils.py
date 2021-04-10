from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

from lhotse.serialization import load_json, load_jsonl, load_yaml, save_to_json, save_to_jsonl, save_to_yaml
from lhotse.utils import TimeSpan, overlaps, overspans


@pytest.mark.parametrize(
    ['lhs', 'rhs', 'expected'], [
        (TimeSpan(0, 1), TimeSpan(0, 1), True),
        (TimeSpan(0.5, 1), TimeSpan(0, 1), True),
        (TimeSpan(0, 0.5), TimeSpan(0, 1), True),
        (TimeSpan(0.1, 0.9), TimeSpan(0, 1), True),
        (TimeSpan(1, 2), TimeSpan(0, 1), False),
        (TimeSpan(2, 3), TimeSpan(0, 1), False),
        (TimeSpan(1, 1), TimeSpan(0, 1), False),
        (TimeSpan(0, 0), TimeSpan(0, 1), False),
        (TimeSpan(-1, 1), TimeSpan(0, 1), True),
        (TimeSpan(0, 1), TimeSpan(-1, 1), True),
        (TimeSpan(-2, -1), TimeSpan(1, 2), False),
    ]
)
def test_overlaps(lhs, rhs, expected):
    assert overlaps(lhs, rhs) == expected
    assert overlaps(rhs, lhs) == expected


@pytest.mark.parametrize(
    ['lhs', 'rhs', 'expected'], [
        (TimeSpan(0, 1), TimeSpan(0, 1), True),
        (TimeSpan(0.5, 1), TimeSpan(0, 1), False),
        (TimeSpan(0, 1), TimeSpan(0.5, 1), True),
        (TimeSpan(0, 0.5), TimeSpan(0, 1), False),
        (TimeSpan(0, 1), TimeSpan(0, 0.5), True),
        (TimeSpan(0.1, 0.9), TimeSpan(0, 1), False),
        (TimeSpan(0, 1), TimeSpan(0.1, 0.9), True),
        (TimeSpan(1, 2), TimeSpan(0, 1), False),
        (TimeSpan(0, 1), TimeSpan(0, 2), False),
        (TimeSpan(2, 3), TimeSpan(0, 1), False),
        (TimeSpan(0, 1), TimeSpan(2, 3), False),
        (TimeSpan(1, 1), TimeSpan(0, 1), False),
        (TimeSpan(0, 1), TimeSpan(1, 1), True),
        (TimeSpan(0, 0), TimeSpan(0, 1), False),
        (TimeSpan(0, 1), TimeSpan(0, 0), True),
        (TimeSpan(-1, 1), TimeSpan(0, 1), True),
        (TimeSpan(0, 1), TimeSpan(-1, 1), False),
        (TimeSpan(-2, -1), TimeSpan(1, 2), False),
    ]
)
def test_overspans(lhs, rhs, expected):
    assert overspans(lhs, rhs) == expected


@pytest.mark.parametrize('extension', ['.yml', '.yml.gz'])
def test_yaml_save_load_roundtrip(extension):
    data = {'some': ['data']}
    with NamedTemporaryFile() as f:
        path = Path(f.name).with_suffix(extension)
        save_to_yaml(data, path)
        f.flush()
        data_deserialized = load_yaml(path)
    assert data == data_deserialized


@pytest.mark.parametrize('extension', ['.json', '.json.gz'])
def test_json_save_load_roundtrip(extension):
    data = {'some': ['data']}
    with NamedTemporaryFile() as f:
        path = Path(f.name).with_suffix(extension)
        save_to_json(data, path)
        f.flush()
        data_deserialized = load_json(path)
    assert data == data_deserialized


@pytest.mark.parametrize('extension', ['.jsonl', '.jsonl.gz'])
def test_jsonl_save_load_roundtrip(extension):
    data = [{'some': ['data']}]
    with NamedTemporaryFile() as f:
        path = Path(f.name).with_suffix(extension)
        save_to_jsonl(data, path)
        f.flush()
        data_deserialized = list(load_jsonl(path))
    assert data == data_deserialized
