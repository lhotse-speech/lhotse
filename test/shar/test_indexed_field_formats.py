"""Exercise Shar storage formats through streaming and both indexed readers."""

import gzip
import io
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest
from packaging.version import parse as parse_version

from lhotse import CutSet, fastcopy
from lhotse.indexing import (
    IndexedTarReader,
    create_jsonl_index,
    index_exists,
    index_file_path,
    read_index,
)
from lhotse.serialization import (
    AIStoreIOBackend,
    BuiltinIOBackend,
    CompositeIOBackend,
    GzipIOBackend,
)
from lhotse.shar.readers.indexed import LazyIndexedSharIterator
from lhotse.shar.readers.lazy import LazySharIterator
from lhotse.shar.writers import SharWriter
from lhotse.testing.dummies import (
    DummyManifest,
    dummy_array,
    dummy_multi_cut,
    dummy_temporal_array_uint8,
)


def _write_fields(
    root, *, audio_format="wav", array_format="numpy", compress=True, include_cuts=True
):
    root.mkdir(parents=True, exist_ok=True)
    cuts = list(DummyManifest(CutSet, begin_id=0, end_id=3, with_data=True))
    fields = {
        "recording": audio_format,
        "features": array_format,
        "custom_embedding": array_format,
        "custom_features": array_format,
        "custom_indexes": "numpy",  # Lossless storage for integer arrays.
        "custom_recording": audio_format,
        "label": "jsonl",
    }
    for cut in cuts:
        cut.label = {"id": cut.id, "languages": ["en", "ja"]}
    # Placeholders exercise absence independently of storage representation.
    cuts[-1].recording = None
    cuts[-1].features = None
    for field in fields.keys() - {"recording", "features"}:
        cuts[-1].custom.pop(field)
    with SharWriter(
        root,
        fields=fields,
        shard_size=2,
        compress_jsonl=compress,
        include_cuts=include_cuts,
    ) as writer:
        for cut in cuts:
            writer.write(cut)
    return cuts, writer.output_paths


def _loaded_fields(cut):
    result = {}
    if cut.has_recording:
        result["recording"] = cut.resample(16000).load_audio()
    if cut.has_features:
        result["features"] = cut.load_features()
    for field in ("custom_embedding", "custom_features", "custom_indexes"):
        if cut.has_custom(field):
            result[field] = cut.load_custom(field)
    if cut.has_custom("custom_recording"):
        result["custom_recording"] = cut.custom_recording.resample(16000).load_audio()
    if cut.has_custom("label"):
        result["label"] = cut.label
    return result


def _assert_fields(actual, expected):
    assert actual.keys() == expected.keys()
    for field in expected:
        if field == "label":
            assert actual[field] == expected[field]
        else:
            np.testing.assert_allclose(actual[field], expected[field], atol=1e-6)


@pytest.mark.parametrize("compress", [False, True])
@pytest.mark.parametrize("array_format", ["numpy", "lilcom"])
@pytest.mark.parametrize("audio_format", ["wav", "flac", "mp3", "opus", "original"])
def test_all_shar_formats_indexed_roundtrip(
    tmp_path, compress, array_format, audio_format
):
    pytest.importorskip("indexed_gzip")
    if array_format == "lilcom":
        pytest.importorskip("lilcom")
    original, paths = _write_fields(
        tmp_path,
        audio_format=audio_format,
        array_format=array_format,
        compress=compress,
    )
    assert all(index_exists(path) for shards in paths.values() for path in shards)
    reference = [_loaded_fields(cut) for cut in LazySharIterator(in_dir=tmp_path)]
    assert reference[-1] == {}
    # Check actual payloads, not only metadata shapes or reader equivalence.
    for cut, fields in zip(original, reference):
        expected = _loaded_fields(cut)
        assert fields.keys() == expected.keys()
        for field in fields.keys() - {"label", "recording", "custom_recording"}:
            np.testing.assert_allclose(fields[field], expected[field], atol=0.04)
        for field in fields.keys() & {"recording", "custom_recording"}:
            assert fields[field].shape == expected[field].shape
            assert np.sqrt(np.mean((fields[field] - expected[field]) ** 2)) < 0.5
    for lazy in (False, True):
        reader = LazyIndexedSharIterator(in_dir=tmp_path, lazy=lazy)
        for position in (2, 1, 0, -1):
            _assert_fields(_loaded_fields(reader[position]), reference[position])


@pytest.mark.parametrize("compress", [False, True])
@pytest.mark.parametrize("source_per_channel", [False, True])
def test_indexed_multichannel_shar_selection_and_truncation(
    tmp_path, compress, source_per_channel
):
    pytest.importorskip("indexed_gzip")
    cut = dummy_multi_cut(0, with_data=True, source_per_channel=source_per_channel)
    cut.custom_recording = cut.recording
    cut.custom_recording_channel_selector = [1]
    cut.custom_indexes = dummy_temporal_array_uint8()
    cut.custom_embedding = dummy_array()
    cut = cut.truncate(offset=0.2, duration=0.6)
    expected_audio = cut.load_audio()
    expected_features = cut.load_features()
    expected_custom = cut.load_custom_recording()
    expected_indexes = cut.load_custom_indexes()
    with SharWriter(
        tmp_path,
        fields={
            "recording": "flac",
            "features": "numpy",
            "custom_recording": "original",
            "custom_indexes": "numpy",
            "custom_embedding": "lilcom",
        },
        compress_jsonl=compress,
        shard_size=None,
    ) as writer:
        writer.write(cut)
    for reader in (
        LazySharIterator(in_dir=tmp_path),
        LazyIndexedSharIterator(in_dir=tmp_path),
        LazyIndexedSharIterator(in_dir=tmp_path, lazy=True),
    ):
        (actual,) = list(reader)
        assert actual.start == 0
        assert actual.channel == [0, 1]
        np.testing.assert_array_equal(actual.load_audio(), expected_audio)
        np.testing.assert_array_equal(actual.load_features(), expected_features)
        np.testing.assert_array_equal(actual.load_custom_indexes(), expected_indexes)
        np.testing.assert_array_equal(actual.load_custom_recording(), expected_custom)
        assert actual.custom_recording.channel_ids == [1]
        np.testing.assert_allclose(
            actual.load_custom_embedding(), cut.load_custom_embedding(), atol=0.04
        )
        shortened = actual.truncate(offset=0.1, duration=0.3)
        np.testing.assert_array_equal(
            shortened.load_audio(), expected_audio[:, 1600:6400]
        )
        np.testing.assert_array_equal(
            shortened.load_custom_indexes(), expected_indexes[10:40]
        )


@pytest.mark.parametrize("compress", [False, True])
@pytest.mark.parametrize("placeholder", [False, True])
def test_indexed_shar_validates_external_jsonl_cut_ids(tmp_path, compress, placeholder):
    pytest.importorskip("indexed_gzip")
    _, paths = _write_fields(tmp_path, compress=compress)
    path = Path(paths["label"][0])
    content = gzip.decompress(path.read_bytes()) if compress else path.read_bytes()
    rows = [json.loads(line) for line in content.splitlines()]
    rows.reverse()
    if placeholder:
        rows = [{"cut_id": row["cut_id"]} for row in rows]
    content = b"".join(json.dumps(row).encode() + b"\n" for row in rows)
    path.write_bytes(gzip.compress(content) if compress else content)
    create_jsonl_index(path)
    for lazy in (False, True):
        reader = LazyIndexedSharIterator(in_dir=tmp_path, lazy=lazy)
        with pytest.raises(AssertionError, match="Mismatched IDs"):
            reader[0]
    with pytest.raises(AssertionError, match="Mismatched IDs"):
        next(iter(CutSet.from_shar(in_dir=tmp_path)))


@pytest.mark.parametrize("compress", [False, True])
def test_lazy_indexed_shar_loads_fields_added_without_rewriting_cuts(
    tmp_path, compress
):
    pytest.importorskip("indexed_gzip")
    with SharWriter(
        tmp_path, fields={}, shard_size=2, compress_jsonl=compress
    ) as writer:
        for cut in DummyManifest(CutSet, begin_id=0, end_id=3):
            writer.write(fastcopy(cut, recording=None, features=None, custom=None))
    _write_fields(
        tmp_path, compress=compress, include_cuts=False, array_format="lilcom"
    )
    reference = [_loaded_fields(cut) for cut in LazySharIterator(in_dir=tmp_path)]
    reader = LazyIndexedSharIterator(in_dir=tmp_path, lazy=True)
    for position in (1, 0, 2):
        actual = reader[position]
        if position < 2:
            assert actual.recording.sources[0].type == "shar_ptr"
            assert actual.features.storage_type == "shar_ptr_array"
            assert actual.custom_embedding.storage_type == "shar_ptr_array"
        _assert_fields(_loaded_fields(actual), reference[position])


@pytest.mark.parametrize("compress", [False, True])
@pytest.mark.parametrize("append_fields", [False, True])
def test_indexed_shar_handles_long_unicode_cut_ids(tmp_path, compress, append_fields):
    pytest.importorskip("indexed_gzip")
    cuts = list(DummyManifest(CutSet, begin_id=0, end_id=2, with_data=True))
    for cut in cuts:
        cut.id = f"nested/{cut.id}-" + "音声" * 70
    if append_fields:
        with SharWriter(
            tmp_path, fields={}, shard_size=None, compress_jsonl=compress
        ) as writer:
            for cut in cuts:
                writer.write(fastcopy(cut, recording=None, features=None, custom=None))
    fields = {
        "recording": "wav",
        "features": "lilcom",
        "custom_embedding": "numpy",
        "custom_features": "lilcom",
        "custom_indexes": "numpy",
        "custom_recording": "flac",
    }
    with SharWriter(
        tmp_path,
        fields=fields,
        shard_size=None,
        compress_jsonl=compress,
        include_cuts=not append_fields,
    ) as writer:
        for cut in cuts:
            writer.write(cut)
    expected = [_loaded_fields(cut) for cut in LazySharIterator(in_dir=tmp_path)]
    for lazy in (False, True):
        reader = LazyIndexedSharIterator(in_dir=tmp_path, lazy=lazy)
        for position in (1, 0):
            actual = reader[position]
            assert actual.id == cuts[position].id
            _assert_fields(_loaded_fields(actual), expected[position])


def test_tar_metadata_access_does_not_fetch_payload(tmp_path, ais_objects):
    pytest.importorskip("indexed_gzip")
    _, paths = _write_fields(tmp_path)
    path = Path(paths["recording"][0])
    objects, requests = ais_objects
    url = f"ais://bucket/{uuid4().hex}/{path.name}"
    objects[url] = path.read_bytes()
    import tarfile

    with tarfile.open(path) as archive:
        data_member = next(iter(archive))
    payload_start = data_member.offset_data
    payload_end = payload_start + data_member.size
    reader = IndexedTarReader(url, index_path=index_file_path(path))
    manifest, member_path = reader.read_metadata(0)
    assert manifest.sources[0].type == "shar"
    assert member_path.stem == "dummy-mono-cut-0000"
    assert requests
    for requested_url, range_ in requests:
        assert requested_url == url
        start, end = map(int, range_.removeprefix("bytes=").split("-"))
        assert end < payload_start or start >= payload_end
    reader.close()


@pytest.fixture(params=["seekable", "streaming"])
def ais_objects(monkeypatch, request):
    """Fake only the SDK transport; exercise real AIS backend and range reader."""
    objects = {}
    requests = []

    class ReadStream(io.BufferedIOBase):
        def __init__(self, data):
            self.data = io.BytesIO(data)

        def read(self, size=-1):
            return self.data.read(size)

        def readable(self):
            return True

    read_stream = io.BytesIO if request.param == "seekable" else ReadStream

    class WriteBuffer(io.BytesIO):
        def __init__(self, url):
            super().__init__()
            self.url = url

        def close(self):
            if not self.closed:
                objects[self.url] = self.getvalue()
            super().close()

    class Object:
        def __init__(self, url):
            self.url = url

        @property
        def props(self):
            return SimpleNamespace(size=len(objects[self.url]))

        def get_reader(self, byte_range=None):
            data = objects[
                self.url
            ]  # Missing objects must raise, never contact a service.
            requests.append((self.url, byte_range))
            if byte_range is not None:
                start, end = map(int, byte_range.removeprefix("bytes=").split("-"))
                assert 0 <= start <= end < len(data)
                data = data[start : end + 1]
            return SimpleNamespace(
                as_file=lambda: read_stream(data), read_all=lambda: data
            )

        def get_writer(self):
            return SimpleNamespace(as_file=lambda: WriteBuffer(self.url))

    client = SimpleNamespace(get_object_from_url=Object)
    monkeypatch.setattr(
        "lhotse.serialization.get_aistore_client",
        lambda: (client, parse_version("1.10.0")),
    )
    monkeypatch.setattr(
        "lhotse.serialization.CURRENT_IO_BACKEND",
        CompositeIOBackend([GzipIOBackend(), AIStoreIOBackend(), BuiltinIOBackend()]),
    )
    from lhotse.shar.lazy_pointer import close_all

    close_all()
    yield objects, requests
    close_all()


@pytest.mark.parametrize("compress", [False, True])
@pytest.mark.parametrize("mirror", [False, True])
def test_shar_ais_all_fields_and_automatic_index_creation(
    tmp_path, ais_objects, compress, mirror
):
    pytest.importorskip("indexed_gzip")
    root = tmp_path / "shar"
    _, paths = _write_fields(root, compress=compress, array_format="lilcom")
    expected = [_loaded_fields(cut) for cut in LazySharIterator(in_dir=root)]
    objects, requests = ais_objects
    namespace = uuid4().hex
    fields = {}
    for field, shards in paths.items():
        fields[field] = []
        for path in shards:
            url = f"ais://bucket/{namespace}/{Path(path).name}"
            objects[url] = Path(path).read_bytes()
            fields[field].append(url)
    indexes_root = tmp_path / "mirror" if mirror else None
    # No indexes are copied. The reader creates both JSONL companions and tar indexes.
    eager = LazyIndexedSharIterator(fields=fields, indexes_root=indexes_root)
    for urls in fields.values():
        for url in urls:
            if url.endswith(".tar"):
                assert int(read_index(index_file_path(url, indexes_root))[-1]) == len(
                    objects[url]
                )
    for position in (2, 0, 1):
        _assert_fields(_loaded_fields(eager[position]), expected[position])
    assert any(url.endswith(".tar") and range_ for url, range_ in requests)
    if compress:
        assert any(url.endswith(".jsonl.gz") and range_ for url, range_ in requests)
    if mirror:
        assert list(indexes_root.rglob("*.idx"))
        assert bool(list(indexes_root.rglob("*.gzidx"))) == compress
        assert not any(url.endswith((".idx", ".gzidx")) for url in objects)
    else:
        assert any(url.endswith(".idx") for url in objects)
        assert any(url.endswith(".gzidx") for url in objects) == compress
    # Prebuilt indexes work in both modes, and lazy construction avoids tar payloads.
    requests.clear()
    lazy = LazyIndexedSharIterator(fields=fields, indexes_root=indexes_root, lazy=True)
    first = lazy[0]
    assert not any(url.endswith(".tar") for url, _ in requests)
    _assert_fields(_loaded_fields(first), expected[0])
    _assert_fields(_loaded_fields(lazy[-1]), expected[-1])
    selected = {key: fields[key] for key in ("cuts", "custom_features", "label")}
    subset = LazyIndexedSharIterator(fields=selected, indexes_root=indexes_root)
    np.testing.assert_allclose(
        subset[1].load_custom_features(), expected[1]["custom_features"]
    )
    assert subset[1].label == expected[1]["label"]
