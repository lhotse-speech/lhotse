from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

from lhotse.serialization import (
    load_json,
    load_jsonl,
    load_yaml,
    save_to_json,
    save_to_jsonl,
    save_to_yaml,
)
from lhotse.utils import (
    TimeSpan,
    add_durations,
    compute_start_duration_for_extended_cut,
    compute_num_windows,
    overlaps,
    overspans,
)


@pytest.mark.parametrize(
    ["lhs", "rhs", "expected"],
    [
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
    ],
)
def test_overlaps(lhs, rhs, expected):
    assert overlaps(lhs, rhs) == expected
    assert overlaps(rhs, lhs) == expected


@pytest.mark.parametrize(
    ["lhs", "rhs", "expected"],
    [
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
    ],
)
def test_overspans(lhs, rhs, expected):
    assert overspans(lhs, rhs) == expected


@pytest.mark.parametrize("extension", [".yml", ".yml.gz"])
def test_yaml_save_load_roundtrip(extension):
    data = {"some": ["data"]}
    with NamedTemporaryFile() as f:
        path = Path(f.name).with_suffix(extension)
        save_to_yaml(data, path)
        f.flush()
        data_deserialized = load_yaml(path)
    assert data == data_deserialized


@pytest.mark.parametrize("extension", [".json", ".json.gz"])
def test_json_save_load_roundtrip(extension):
    data = {"some": ["data"]}
    with NamedTemporaryFile() as f:
        path = Path(f.name).with_suffix(extension)
        save_to_json(data, path)
        f.flush()
        data_deserialized = load_json(path)
    assert data == data_deserialized


@pytest.mark.parametrize("extension", [".jsonl", ".jsonl.gz"])
def test_jsonl_save_load_roundtrip(extension):
    data = [{"some": ["data"]}]
    with NamedTemporaryFile() as f:
        path = Path(f.name).with_suffix(extension)
        save_to_jsonl(data, path)
        f.flush()
        data_deserialized = list(load_jsonl(path))
    assert data == data_deserialized


@pytest.mark.parametrize(
    ["direction", "expected_start"],
    [("left", 4.0), ("right", 5.0), ("center", 4.5), ("random", None)],
)
def test_compute_start_when_extending_duration(direction, expected_start):
    # noinspection PyTypeChecker
    new_start, new_duration = compute_start_duration_for_extended_cut(
        start=5.0,
        duration=10.0,
        new_duration=11.0,
        direction=direction,
    )
    assert new_duration == 11.0
    if direction != "random":
        assert new_start == expected_start
    else:
        assert 4.0 <= new_start <= 5.0


def test_compute_start_when_extending_duration_exceeded_left_side():
    # noinspection PyTypeChecker
    new_start, new_duration = compute_start_duration_for_extended_cut(
        start=0.0,
        duration=10.0,
        new_duration=11.0,
        direction="center",
    )
    assert new_start == 0.0
    assert new_duration == 10.5


def test_compute_start_when_extending_duration_incorrect_direction():
    with pytest.raises(ValueError):
        compute_start_duration_for_extended_cut(0, 1, 2, "bad-direction-string")


def test_add_durations():
    begin = 0.94
    end = 7.06
    expected = 6.12
    naive = end - begin
    assert naive != expected
    safe = add_durations(end, -begin, sampling_rate=16000)
    assert safe == expected


@pytest.mark.parametrize(
    "params, expected_n_win",  # params: (sig_len,win_len,hop)
    [
        ((1, 6.1, 3), 1),  # 0-1
        ((3, 1, 6.1), 1),  # 0-1
        ((3, 6.1, 1), 1),  # 0-3
        ((5.9, 1, 3), 2),  # 0-1, 3-4
        ((5.9, 3, 1), 4),  # 0-3, 1-4, 2-5, 3-5.9
        ((6.1, 1, 3), 3),  # 0-1, 3-4, 6-6.1
        ((6.1, 3, 1), 5),  # 0-3, 1-4, 2-5, 3-6, 4-6.1
        ((5.9, 3, 3), 2),  # 0-3, 3-5.9
        ((6.1, 3, 3), 3),  # 0-3, 3-6, 6-6.1
        ((0.0, 3, 3), 0),
    ],
)
def test_compute_num_windows(params, expected_n_win):
    assert compute_num_windows(params[0], params[1], params[2]) == expected_n_win
