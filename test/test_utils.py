import tarfile
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import click
import pytest
from click.testing import CliRunner

from lhotse.serialization import (
    load_json,
    load_jsonl,
    load_yaml,
    save_to_json,
    save_to_jsonl,
    save_to_yaml,
)
from lhotse.utils import (
    PythonLiteralOption,
    TimeSpan,
    add_durations,
    compute_num_windows,
    compute_start_duration_for_extended_cut,
    overlaps,
    overspans,
    safe_extract,
    safe_extract_rar,
    streaming_shuffle,
)


@pytest.fixture
def safe_tar_file():
    with NamedTemporaryFile() as f:
        with tarfile.open(f.name, "w:gz") as tar:
            tar.add("test/fixtures/audio.json")
        yield f.name


@pytest.fixture
def unsafe_tar_file():
    def _change_name(tarinfo):
        tarinfo.name = "../" + tarinfo.name
        return tarinfo

    with NamedTemporaryFile() as f:
        with tarfile.open(f.name, "w:gz") as tar:
            tar.add("test/fixtures/audio.json", filter=_change_name)
        yield f.name


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


@pytest.mark.parametrize(
    ["input_size", "bufsize", "expected"],
    [
        (0, 0, True),
        (0, 1, True),
        (1, 0, True),
        (1, 1, True),
        (1, 10000, True),
        (32, 0, True),
        (32, 1, False),
        (32, 10000, False),
    ],
)
def test_streaming_shuffle(input_size, bufsize, expected):
    input = range(input_size)
    output = [x for x in streaming_shuffle(iter(input), bufsize)]
    assert len(list(input)) == len(output)
    assert expected == (list(input) == output)


def test_extract_safe_tar_file(safe_tar_file):
    with TemporaryDirectory() as tmpdir, tarfile.open(safe_tar_file) as tar:
        safe_extract(tar, path=tmpdir)
        assert (Path(tmpdir) / "test/fixtures/audio.json").is_file()


def test_extract_unsafe_tar_file(unsafe_tar_file):
    with TemporaryDirectory() as tmpdir, tarfile.open(unsafe_tar_file) as tar:
        with pytest.raises(Exception):
            safe_extract(tar, tmpdir)


# rarfile has no create archive implementation, so for testing purposes, present a TarFile as a RarFile
class TarInfo2RarInfo:
    def __init__(self, tarinfo):
        self.tarinfo = tarinfo
        self.filename = tarinfo.name


class TarFile2RarFile:
    def __init__(self, tar):
        self.tar = tar

    def infolist(self):
        return [TarInfo2RarInfo(m) for m in self.tar.getmembers()]

    def extractall(self, path, members):
        return self.tar.extractall(path, members)


def test_extract_safe_rar_file(safe_tar_file):
    with TemporaryDirectory() as tmpdir, tarfile.open(safe_tar_file) as tar:
        safe_extract_rar(TarFile2RarFile(tar), path=tmpdir)
        assert (Path(tmpdir) / "test/fixtures/audio.json").is_file()


def test_extract_unsafe_rar_file(unsafe_tar_file):
    with TemporaryDirectory() as tmpdir, tarfile.open(unsafe_tar_file) as tar:
        with pytest.raises(Exception):
            safe_extract_rar(TarFile2RarFile(tar), tmpdir)


@pytest.mark.parametrize(
    ["value", "expected"], [(2, None), ("3", 3), ("(4, 5)", (4, 5)), ("[6, 7]", [6, 7])]
)
def test_click_literal_option(value, expected):
    @click.command()
    @click.option("--num", "-n", cls=PythonLiteralOption, default=2)
    def echo(num):
        click.echo("Value: {}".format(num))

    runner = CliRunner()
    result = runner.invoke(echo, ["-n", value])
    assert result.output == "Value: {}\n".format(expected)
