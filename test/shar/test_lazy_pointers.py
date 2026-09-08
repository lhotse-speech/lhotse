"""
Tests for the Lhotse Shar lazy-pointer mode (``LazyIndexedSharIterator(lazy=True)``)
and the supporting ``shar_ptr`` / ``shar_ptr_array`` storage types.
"""

from __future__ import annotations

import pickle
import re
import tarfile
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

# Reuse the standard shar test fixture which specifies all fields
from test.shar.conftest import cuts  # noqa: F401
from threading import Barrier
from unittest.mock import patch

import numpy as np
import pytest

from lhotse.shar.lazy_pointer import (
    close_all,
    decode_pointer,
    decode_pointer_with_name,
    encode_pointer,
    is_shar_pointer,
)
from lhotse.shar.readers.indexed import LazyIndexedSharIterator
from lhotse.shar.writers.shar import SharWriter

ALL_FIELDS_LILCOM = {
    "recording": "wav",
    "features": "lilcom",
    "custom_embedding": "numpy",
    "custom_features": "numpy",
    "custom_indexes": "numpy",
    "custom_recording": "wav",
}

ALL_FIELDS_NUMPY = {
    "recording": "wav",
    "features": "numpy",
    "custom_embedding": "numpy",
    "custom_features": "numpy",
    "custom_indexes": "numpy",
    "custom_recording": "wav",
}


@pytest.fixture
def shar_dir_lilcom(tmp_path, cuts):
    """An indexed Shar dir using lilcom for ``features`` (numpy for arrays)."""
    pytest.importorskip("lilcom")
    writer = SharWriter(
        tmp_path,
        fields=ALL_FIELDS_LILCOM,
        shard_size=10,
        compress_jsonl=False,
        create_index=True,
    )
    with writer:
        for c in cuts:
            writer.write(c)
    yield tmp_path
    close_all()


@pytest.fixture
def shar_dir_numpy(tmp_path, cuts):
    """An indexed Shar dir using numpy for every array-shaped field."""
    writer = SharWriter(
        tmp_path,
        fields=ALL_FIELDS_NUMPY,
        shard_size=10,
        compress_jsonl=False,
        create_index=True,
    )
    with writer:
        for c in cuts:
            writer.write(c)
    yield tmp_path
    close_all()


# ---------------------------------------------------------------------------
# Wire-format helpers
# ---------------------------------------------------------------------------


def test_encode_decode_pointer_roundtrip():
    p = encode_pointer("/some/where.tar", 1024, 65536)
    assert p == "/some/where.tar?o=1024&e=65536"
    assert decode_pointer(p) == ("/some/where.tar", 1024, 65536)


def test_encode_pointer_with_expected_member_name_is_backward_compatible():
    p = encode_pointer(
        "/some/where.tar",
        1024,
        65536,
        expected_name="nested/audio with spaces&symbols.wav",
    )

    assert p == (
        "/some/where.tar?o=1024&e=65536&n=nested%2Faudio%20with%20spaces%26symbols.wav"
    )
    assert decode_pointer(p) == ("/some/where.tar", 1024, 65536)
    assert decode_pointer_with_name(p) == (
        "/some/where.tar",
        1024,
        65536,
        "nested/audio with spaces&symbols.wav",
    )
    assert is_shar_pointer(p)


def test_encode_strict_pointer_requires_and_preserves_expected_name():
    p = encode_pointer(
        "/some/where.tar",
        1024,
        65536,
        expected_name="audio.wav",
        strict=True,
    )

    assert p == "/some/where.tar?o=1024&e=65536&n=audio.wav&s=1"
    assert decode_pointer(p) == ("/some/where.tar", 1024, 65536)
    assert decode_pointer_with_name(p) == (
        "/some/where.tar",
        1024,
        65536,
        "audio.wav",
    )
    assert is_shar_pointer(p)
    with pytest.raises(ValueError, match="require expected_name"):
        encode_pointer("/some/where.tar", 0, 1, strict=True)


def test_expected_member_name_roundtrips_surrogateescaped_bytes():
    name = "audio-\udcff.wav"
    pointer = encode_pointer("/some/where.tar", 0, 1024, expected_name=name)

    assert decode_pointer_with_name(pointer)[3] == name


def test_is_shar_pointer():
    assert is_shar_pointer("/x.tar?o=0&e=10")
    assert not is_shar_pointer("/x.tar")
    assert not is_shar_pointer("/x.tar?o=foo&e=10")
    assert not is_shar_pointer(b"/x.tar?o=0&e=10")  # not a str


def test_decode_pointer_rejects_malformed():
    for bad in ("garbage", "/x.tar?o=10", "/x.tar?o=10&e=20&extra=1"):
        with pytest.raises(ValueError):
            decode_pointer(bad)


def _tar_member_ranges(path: Path) -> dict[str, tuple[int, int]]:
    with tarfile.open(path, "r:") as archive:
        members = [member for member in archive if member.isfile()]
    return {
        member.name: (
            member.offset,
            members[idx + 1].offset if idx + 1 < len(members) else path.stat().st_size,
        )
        for idx, member in enumerate(members)
    }


def test_read_payload_validates_member_name_while_loading(tmp_path):
    tar_path = tmp_path / "audio.tar"
    payloads = {"first.wav": b"first-audio", "second.wav": b"second-audio"}
    with tarfile.open(tar_path, "w") as archive:
        for name, payload in payloads.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, BytesIO(payload))
    ranges = _tar_member_ranges(tar_path)
    start, end = ranges["first.wav"]

    pointer = encode_pointer(tar_path, start, end, expected_name="first.wav")

    from lhotse.shar.lazy_pointer import read_payload

    assert read_payload(pointer) == payloads["first.wav"]


def test_read_payload_does_not_read_trailing_shar_metadata(tmp_path, monkeypatch):
    from lhotse.shar import lazy_pointer

    tar_path = tmp_path / "recording.tar"
    audio = b"audio"
    metadata = b"x" * (1024 * 1024)
    with tarfile.open(tar_path, "w") as archive:
        for name, payload in (
            ("sample.wav", audio),
            ("sample.json", metadata),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, BytesIO(payload))

    class CountingFile:
        def __init__(self, path):
            self.file = open(path, "rb")
            self.bytes_read = 0

        def read(self, size=-1):
            data = self.file.read(size)
            self.bytes_read += len(data)
            return data

        def seek(self, *args):
            return self.file.seek(*args)

        def tell(self):
            return self.file.tell()

        def close(self):
            self.file.close()

    opened = []

    def counting_open_best(path, mode):
        handle = CountingFile(path)
        opened.append(handle)
        return handle

    close_all()
    monkeypatch.setattr(lazy_pointer, "open_best", counting_open_best)
    pointer = encode_pointer(
        tar_path,
        0,
        tar_path.stat().st_size,
        expected_name="sample.wav",
    )

    assert lazy_pointer.read_payload(pointer) == audio
    assert len(opened) == 1
    assert opened[0].bytes_read < len(metadata)


def test_read_payload_resolves_filtered_manifest_member_on_first_mismatch(tmp_path):
    tar_path = tmp_path / "audio.tar"
    payloads = {
        "discarded.wav": b"discarded-audio",
        "selected.wav": b"selected-audio",
    }
    with tarfile.open(tar_path, "w") as archive:
        for name, payload in payloads.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, BytesIO(payload))
    ranges = _tar_member_ranges(tar_path)
    candidate_start, candidate_end = ranges["discarded.wav"]

    pointer = encode_pointer(
        tar_path,
        candidate_start,
        candidate_end,
        expected_name="selected.wav",
    )

    from lhotse.shar.lazy_pointer import read_payload

    assert read_payload(pointer) == payloads["selected.wav"]


def test_read_payload_strict_name_mismatch_never_scans_tar(tmp_path):
    from lhotse.audio import AudioLoadingError
    from lhotse.shar import lazy_pointer

    tar_path = tmp_path / "audio.tar"
    payloads = {
        "discarded.wav": b"discarded-audio",
        "selected.wav": b"selected-audio",
    }
    with tarfile.open(tar_path, "w") as archive:
        for name, payload in payloads.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, BytesIO(payload))
    candidate_range = _tar_member_ranges(tar_path)["discarded.wav"]
    selected_range = _tar_member_ranges(tar_path)["selected.wav"]
    pointer = encode_pointer(
        tar_path,
        *candidate_range,
        expected_name="selected.wav",
        strict=True,
    )
    matching_pointer = encode_pointer(
        tar_path,
        *selected_range,
        expected_name="selected.wav",
        strict=True,
    )

    close_all()
    with patch.object(
        lazy_pointer,
        "_build_member_index",
        side_effect=AssertionError("strict pointers must not scan the tar"),
    ) as build_member_index:
        assert lazy_pointer.read_payload(matching_pointer) == payloads["selected.wav"]
        with pytest.raises(AudioLoadingError, match="name mismatch"):
            lazy_pointer.read_payload(pointer)
    build_member_index.assert_not_called()


def test_read_payload_strict_pointer_ignores_cached_recovery_index(tmp_path):
    from lhotse.audio import AudioLoadingError
    from lhotse.shar import lazy_pointer

    tar_path = tmp_path / "audio.tar"
    payloads = {"first.wav": b"first", "second.wav": b"second"}
    with tarfile.open(tar_path, "w") as archive:
        for name, payload in payloads.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, BytesIO(payload))
    first_range = _tar_member_ranges(tar_path)["first.wav"]
    recovering = encode_pointer(tar_path, *first_range, expected_name="second.wav")
    strict = encode_pointer(
        tar_path,
        *first_range,
        expected_name="second.wav",
        strict=True,
    )

    close_all()
    assert lazy_pointer.read_payload(recovering) == payloads["second.wav"]
    with pytest.raises(AudioLoadingError, match="name mismatch"):
        lazy_pointer.read_payload(strict)


def test_read_payload_reuses_filtered_manifest_name_index(tmp_path):
    from lhotse.shar import lazy_pointer

    tar_path = tmp_path / "audio.tar"
    payloads = {
        "discarded.wav": b"discarded-audio",
        "selected.wav": b"selected-audio",
    }
    with tarfile.open(tar_path, "w") as archive:
        for name, payload in payloads.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, BytesIO(payload))
    candidate_range = _tar_member_ranges(tar_path)["discarded.wav"]
    pointer = encode_pointer(
        tar_path,
        *candidate_range,
        expected_name="selected.wav",
    )

    close_all()
    assert lazy_pointer.read_payload(pointer) == payloads["selected.wav"]
    with patch.object(
        lazy_pointer,
        "_read_first_regular_member",
        side_effect=AssertionError("cached names must bypass the candidate range"),
    ):
        assert lazy_pointer.read_payload(pointer) == payloads["selected.wav"]


def test_read_payload_reports_missing_expected_member_as_audio_error(tmp_path):
    from lhotse.audio import AudioLoadingError
    from lhotse.shar.lazy_pointer import read_payload

    tar_path = tmp_path / "audio.tar"
    with tarfile.open(tar_path, "w") as archive:
        payload = b"audio"
        info = tarfile.TarInfo("present.wav")
        info.size = len(payload)
        archive.addfile(info, BytesIO(payload))
    start, end = _tar_member_ranges(tar_path)["present.wav"]
    pointer = encode_pointer(
        tar_path,
        start,
        end,
        expected_name="missing.wav",
    )

    with pytest.raises(AudioLoadingError, match="no member named 'missing.wav'"):
        read_payload(pointer)


def test_read_payload_rejects_corrupt_candidate_range_without_scanning(tmp_path):
    from lhotse.audio import AudioLoadingError
    from lhotse.shar import lazy_pointer

    tar_path = tmp_path / "audio.tar"
    payloads = {"first.wav": b"first-audio", "second.wav": b"second-audio"}
    with tarfile.open(tar_path, "w") as archive:
        for name, payload in payloads.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, BytesIO(payload))
    candidate_end = tar_path.stat().st_size
    pointer = encode_pointer(
        tar_path,
        candidate_end - 1,
        candidate_end,
        expected_name="second.wav",
    )

    close_all()
    with patch.object(
        lazy_pointer,
        "_build_member_index",
        wraps=lazy_pointer._build_member_index,
    ) as build_member_index:
        with pytest.raises(AudioLoadingError):
            lazy_pointer.read_payload(pointer)
    build_member_index.assert_not_called()


def test_read_payload_preserves_malformed_pointer_error():
    from lhotse.shar.lazy_pointer import read_payload

    with pytest.raises(ValueError, match="Not a Shar pointer"):
        read_payload("not-a-pointer")


@pytest.mark.parametrize("tar_format", [tarfile.PAX_FORMAT, tarfile.GNU_FORMAT])
def test_read_payload_parses_extended_name_from_indexed_range(tmp_path, tar_format):
    tar_path = tmp_path / "audio.tar"
    member_name = f"nested/{'long-' * 24}audio.wav"
    payload = b"pax-audio"
    with tarfile.open(tar_path, "w", format=tar_format) as archive:
        info = tarfile.TarInfo(member_name)
        info.size = len(payload)
        archive.addfile(info, BytesIO(payload))
    start, end = _tar_member_ranges(tar_path)[member_name]

    pointer = encode_pointer(tar_path, start, end, expected_name=member_name)

    from lhotse.shar.lazy_pointer import read_payload

    assert read_payload(pointer) == payload


def test_read_payload_resolves_s3_pointer_to_local_mirror_once(tmp_path, monkeypatch):
    from lhotse.shar import lazy_pointer

    tar_path = tmp_path / "bucket" / "key" / "audio.tar"
    tar_path.parent.mkdir(parents=True)
    payload = b"mirrored-audio"
    with tarfile.open(tar_path, "w") as archive:
        info = tarfile.TarInfo("audio.wav")
        info.size = len(payload)
        archive.addfile(info, BytesIO(payload))
    start, end = _tar_member_ranges(tar_path)["audio.wav"]
    pointer = encode_pointer(
        "s3://bucket/key/audio.tar",
        start,
        end,
        expected_name="audio.wav",
    )
    opened = []
    real_open_best = lazy_pointer.open_best

    def recording_open_best(path, mode):
        opened.append(str(path))
        return real_open_best(path, mode)

    close_all()
    monkeypatch.setenv("LHOTSE_S3_LOCAL_MIRROR_ROOTS", str(tmp_path))
    monkeypatch.setattr(lazy_pointer, "open_best", recording_open_best)

    assert lazy_pointer.read_payload(pointer) == payload
    assert lazy_pointer.read_payload(pointer) == payload
    assert opened == [str(tar_path)]


def test_read_payload_is_thread_safe_for_shared_tar(tmp_path):
    from lhotse.shar.lazy_pointer import read_payload

    tar_path = tmp_path / "audio.tar"
    payloads = {f"audio-{idx}.wav": bytes([idx]) * 100 for idx in range(4)}
    with tarfile.open(tar_path, "w") as archive:
        for name, payload in payloads.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, BytesIO(payload))
    ranges = _tar_member_ranges(tar_path)
    pointers = [
        encode_pointer(tar_path, *ranges[name], expected_name=name) for name in payloads
    ]

    with ThreadPoolExecutor(max_workers=8) as executor:
        actual = list(executor.map(read_payload, pointers * 10))

    assert actual == [payloads[name] for name in payloads] * 10


def test_opening_different_archives_is_not_serialized(tmp_path, monkeypatch):
    from lhotse.shar import lazy_pointer

    pointers = []
    for idx in range(2):
        tar_path = tmp_path / f"audio-{idx}.tar"
        payload = bytes([idx])
        with tarfile.open(tar_path, "w") as archive:
            info = tarfile.TarInfo(f"audio-{idx}.wav")
            info.size = len(payload)
            archive.addfile(info, BytesIO(payload))
        pointers.append(
            encode_pointer(
                tar_path,
                *_tar_member_ranges(tar_path)[f"audio-{idx}.wav"],
                expected_name=f"audio-{idx}.wav",
            )
        )

    barrier = Barrier(2)
    real_open_best = lazy_pointer.open_best

    def synchronized_open(path, mode):
        barrier.wait(timeout=2)
        return real_open_best(path, mode)

    close_all()
    monkeypatch.setattr(lazy_pointer, "open_best", synchronized_open)
    with ThreadPoolExecutor(max_workers=2) as executor:
        assert list(executor.map(lazy_pointer.read_payload, pointers)) == [b"\0", b"\1"]


def test_read_payload_evicts_unused_handles(tmp_path, monkeypatch):
    from lhotse.shar import lazy_pointer

    pointers = []
    for idx in range(2):
        tar_path = tmp_path / f"audio-{idx}.tar"
        with tarfile.open(tar_path, "w") as archive:
            payload = bytes([idx])
            info = tarfile.TarInfo(f"audio-{idx}.wav")
            info.size = len(payload)
            archive.addfile(info, BytesIO(payload))
        start, end = _tar_member_ranges(tar_path)[f"audio-{idx}.wav"]
        pointers.append(
            encode_pointer(
                tar_path,
                start,
                end,
                expected_name=f"audio-{idx}.wav",
            )
        )

    close_all()
    monkeypatch.setattr(lazy_pointer, "_MAX_OPEN_FILES", 1)

    assert lazy_pointer.read_payload(pointers[0]) == b"\0"
    first_entry = next(iter(lazy_pointer._HANDLES.values()))
    assert lazy_pointer.read_payload(pointers[1]) == b"\1"
    assert first_entry.handle.closed
    assert len(lazy_pointer._HANDLES) == 1
    assert list(lazy_pointer._HANDLES) == [str(tmp_path / "audio-1.tar")]
    assert lazy_pointer.read_payload(pointers[0]) == b"\0"


def test_close_all_defers_closing_an_active_handle():
    from lhotse.shar import lazy_pointer

    handle = BytesIO(b"tar")
    entry = lazy_pointer._HandleEntry(
        handle=handle,
        lock=lazy_pointer.threading.Lock(),
        users=1,
    )
    close_all()
    lazy_pointer._HANDLES["archive.tar"] = entry

    close_all()

    assert not handle.closed
    assert entry.close_requested
    lazy_pointer._release_handle(entry)
    assert handle.closed
    assert "archive.tar" not in lazy_pointer._HANDLES


# ---------------------------------------------------------------------------
# Lazy mode: zero tar reads at iter time
# ---------------------------------------------------------------------------


def test_lazy_mode_does_not_read_tars_at_iter_time(shar_dir_numpy):
    """Iterating in lazy mode must not consume tar payload bytes."""
    from lhotse.indexing import IndexedTarReader

    real_getitem = IndexedTarReader.__getitem__
    real_read_member = IndexedTarReader._read_member
    counts = {"getitem": 0, "read_member": 0}

    def counting_getitem(self, idx):
        counts["getitem"] += 1
        return real_getitem(self, idx)

    def counting_read_member(self, offset):
        counts["read_member"] += 1
        return real_read_member(self, offset)

    with patch.object(IndexedTarReader, "__getitem__", counting_getitem), patch.object(
        IndexedTarReader, "_read_member", counting_read_member
    ):
        it = LazyIndexedSharIterator(in_dir=shar_dir_numpy, lazy=True)
        items = list(it)

    assert len(items) == 20
    # Lazy mode bypasses IndexedTarReader's eager paths entirely.
    assert counts["getitem"] == 0
    assert counts["read_member"] == 0

    # Sanity: the cuts carry pointer-typed sources / storage.
    sample = items[0]
    assert sample.recording.sources[0].type == "shar_ptr"
    assert is_shar_pointer(sample.recording.sources[0].source)
    assert sample.features.storage_type == "shar_ptr_array"
    assert is_shar_pointer(sample.features.storage_key)


# ---------------------------------------------------------------------------
# Lazy vs eager parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", ["shar_dir_numpy", "shar_dir_lilcom"])
def test_lazy_load_audio_matches_eager(request, fixture):
    shar_dir = request.getfixturevalue(fixture)
    eager = list(LazyIndexedSharIterator(in_dir=shar_dir, lazy=False))
    lazy = list(LazyIndexedSharIterator(in_dir=shar_dir, lazy=True))
    assert len(eager) == len(lazy) == 20
    eager_by_id = {c.id: c for c in eager}
    for c in lazy:
        a = c.load_audio()
        b = eager_by_id[c.id].load_audio()
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("fixture", ["shar_dir_numpy", "shar_dir_lilcom"])
def test_lazy_load_features_matches_eager(request, fixture):
    shar_dir = request.getfixturevalue(fixture)
    eager = list(LazyIndexedSharIterator(in_dir=shar_dir, lazy=False))
    lazy = list(LazyIndexedSharIterator(in_dir=shar_dir, lazy=True))
    eager_by_id = {c.id: c for c in eager}
    for c in lazy:
        a = c.load_features()
        b = eager_by_id[c.id].load_features()
        np.testing.assert_allclose(a, b, rtol=0, atol=0)


def test_lazy_load_custom_arrays_matches_eager(shar_dir_numpy):
    eager = list(LazyIndexedSharIterator(in_dir=shar_dir_numpy, lazy=False))
    lazy = list(LazyIndexedSharIterator(in_dir=shar_dir_numpy, lazy=True))
    eager_by_id = {c.id: c for c in eager}
    for c in lazy:
        # custom_embedding is an Array, custom_features/custom_indexes are TemporalArrays
        for field in ("custom_embedding", "custom_features", "custom_indexes"):
            np.testing.assert_array_equal(
                getattr(c, f"load_{field}")(),
                getattr(eager_by_id[c.id], f"load_{field}")(),
            )


def test_lazy_load_custom_recording_matches_eager(shar_dir_numpy):
    eager = list(LazyIndexedSharIterator(in_dir=shar_dir_numpy, lazy=False))
    lazy = list(LazyIndexedSharIterator(in_dir=shar_dir_numpy, lazy=True))
    eager_by_id = {c.id: c for c in eager}
    for c in lazy:
        # custom_recording is a custom audio field — also routed via shar_ptr.
        np.testing.assert_array_equal(
            c.load_custom_recording(), eager_by_id[c.id].load_custom_recording()
        )


# ---------------------------------------------------------------------------
# JSON round-trip: no bytes should appear in the dict form
# ---------------------------------------------------------------------------


def test_json_roundtrip_lazy_cut_carries_no_bytes(shar_dir_numpy):
    from lhotse.serialization import deserialize_item

    it = LazyIndexedSharIterator(in_dir=shar_dir_numpy, lazy=True)
    c = next(iter(it))
    d = c.to_dict()

    # Walk the dict and verify no bytes anywhere — strings only.
    def assert_no_bytes(obj, path="<root>"):
        if isinstance(obj, bytes):
            raise AssertionError(f"Found bytes at {path}")
        if isinstance(obj, dict):
            for k, v in obj.items():
                assert_no_bytes(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                assert_no_bytes(v, f"{path}[{i}]")

    assert_no_bytes(d)

    c2 = deserialize_item(d)
    np.testing.assert_array_equal(c2.load_audio(), c.load_audio())


# ---------------------------------------------------------------------------
# Audio format inference from pointer bytes
# ---------------------------------------------------------------------------


def test_audio_format_inferred_from_payload(shar_dir_numpy):
    it = LazyIndexedSharIterator(in_dir=shar_dir_numpy, lazy=True)
    c = next(iter(it))
    # Recording field was written with format='wav'.
    fmt = c.recording.sources[0].format
    assert fmt == "wav", f"expected wav, got {fmt!r}"


# ---------------------------------------------------------------------------
# AIS forward-compat scaffold
# ---------------------------------------------------------------------------


def _aistore_at_least(aistore, version: tuple[int, int, int]) -> bool:
    m = re.match(r"^(\d+)\.(\d+)\.(\d+)", getattr(aistore, "__version__", ""))
    return m is not None and tuple(map(int, m.groups())) >= version


def test_ais_byte_range_support_follows_sdk_schema():
    """``aistore>=1.25.0`` exposes byte ranges through the MOSS request schema."""
    aistore = pytest.importorskip("aistore")
    from lhotse.ais.batch_loader import AISBatchLoader

    assert AISBatchLoader._aistore_byte_range_supported() is _aistore_at_least(
        aistore, (1, 25, 0)
    )


def test_ais_collect_queues_shar_ptr_fallback_when_byte_range_unsupported():
    """When the SDK can't do byte ranges, the Shar pointer is queued in
    ``shar_ptr_fallback`` (drained by :meth:`AISBatchLoader.__call__` via
    per-object byte-range gets) — the batch itself stays untouched, and
    :meth:`_collect_manifest_urls` reports success because the pointer was
    scheduled (just on the fallback leg, not the batch leg)."""
    pytest.importorskip("aistore")
    from lhotse import AudioSource, Recording
    from lhotse.ais.batch_loader import AISBatchLoader

    rec = Recording(
        id="x",
        sources=[
            AudioSource(
                type="shar_ptr",
                channels=[0],
                source=encode_pointer("ais://b/recording.000000.tar", 1024, 8192),
            )
        ],
        sampling_rate=16000,
        num_samples=16000,
        duration=1.0,
    )

    with patch.object(
        AISBatchLoader, "_aistore_byte_range_supported", staticmethod(lambda: False)
    ):
        loader = AISBatchLoader.__new__(AISBatchLoader)  # bypass __init__
        loader._client = None  # not touched on this code path
        batch = []
        shar_ptr_fallback = []
        result = loader._collect_manifest_urls(
            rec,
            batch,
            shar_ptr_uses_batch=False,
            shar_ptr_fallback=shar_ptr_fallback,
            manifest_idx=0,
        )

    assert result is True
    assert batch == []
    assert len(shar_ptr_fallback) == 1
    manifest_idx, bck_name, provider, obj_name, offset, length = shar_ptr_fallback[0]
    assert manifest_idx == 0
    assert bck_name == "b"
    assert obj_name == "recording.000000.tar"
    assert offset == 1024
    assert length == 8192 - 1024


def test_ais_does_not_batch_name_validated_pointer():
    """Name-aware pointers stay on the path that can validate and recover."""
    pytest.importorskip("aistore")
    from lhotse import AudioSource, Recording
    from lhotse.ais.batch_loader import AISBatchLoader

    rec = Recording(
        id="x",
        sources=[
            AudioSource(
                type="shar_ptr",
                channels=[0],
                source=encode_pointer(
                    "ais://b/recording.000000.tar",
                    1024,
                    8192,
                    expected_name="wanted.wav",
                ),
            )
        ],
        sampling_rate=16000,
        num_samples=16000,
        duration=1.0,
    )
    loader = AISBatchLoader.__new__(AISBatchLoader)
    shar_ptr_fallback = []

    assert not loader._collect_manifest_urls(
        rec,
        object(),
        shar_ptr_uses_batch=True,
        shar_ptr_fallback=shar_ptr_fallback,
        manifest_idx=0,
    )
    assert shar_ptr_fallback == []


def test_name_validated_ais_pointer_uses_seekable_range_reader(tmp_path, monkeypatch):
    pytest.importorskip("aistore")
    from lhotse import AudioSource, CutSet, Recording
    from lhotse.ais.batch_loader import AISBatchLoader
    from lhotse.shar.lazy_pointer import read_payload

    tar_path = tmp_path / "audio.tar"
    payloads = {"discarded.wav": b"discarded", "selected.wav": b"selected"}
    with tarfile.open(tar_path, "w") as archive:
        for name, payload in payloads.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, BytesIO(payload))
    ranges = _tar_member_ranges(tar_path)
    tar_bytes = tar_path.read_bytes()
    candidate_start, candidate_end = ranges["discarded.wav"]
    pointer = encode_pointer(
        "ais://bucket/audio.tar",
        candidate_start,
        candidate_end,
        expected_name="selected.wav",
    )
    recording = Recording(
        id="x",
        sources=[AudioSource(type="shar_ptr", channels=[0], source=pointer)],
        sampling_rate=16000,
        num_samples=16000,
        duration=1.0,
    )
    loader = AISBatchLoader.__new__(AISBatchLoader)
    opened = []

    def fake_range_reader(path):
        opened.append(path)
        return BytesIO(tar_bytes)

    close_all()
    monkeypatch.setattr("lhotse.ais.AISRangeReader", fake_range_reader)

    assert not loader._cuts_have_ais_data(CutSet.from_cuts([recording.to_cut()]))
    assert read_payload(pointer) == payloads["selected.wav"]
    assert opened == ["ais://bucket/audio.tar"]


def test_s3_pointer_uses_range_reader_with_active_aistore_backend(
    tmp_path, monkeypatch
):
    from lhotse.serialization import AIStoreIOBackend, io_backend
    from lhotse.shar import lazy_pointer

    tar_path = tmp_path / "audio.tar"
    payload = b"audio"
    with tarfile.open(tar_path, "w") as archive:
        info = tarfile.TarInfo("audio.wav")
        info.size = len(payload)
        archive.addfile(info, BytesIO(payload))
    start, end = _tar_member_ranges(tar_path)["audio.wav"]
    pointer = encode_pointer(
        "s3://bucket/audio.tar",
        start,
        end,
        expected_name="audio.wav",
    )
    opened = []

    def fake_range_reader(path):
        opened.append(path)
        return BytesIO(tar_path.read_bytes())

    close_all()
    monkeypatch.setattr("lhotse.ais.AISRangeReader", fake_range_reader)
    with io_backend(AIStoreIOBackend()):
        assert lazy_pointer.read_payload(pointer) == payload
    assert opened == ["s3://bucket/audio.tar"]


def test_non_ais_url_pointer_retains_open_best_backend(tmp_path, monkeypatch):
    from lhotse.shar import lazy_pointer

    tar_path = tmp_path / "audio.tar"
    payload = b"audio"
    with tarfile.open(tar_path, "w") as archive:
        info = tarfile.TarInfo("audio.wav")
        info.size = len(payload)
        archive.addfile(info, BytesIO(payload))
    start, end = _tar_member_ranges(tar_path)["audio.wav"]
    pointer = encode_pointer(
        "https://example.com/audio.tar",
        start,
        end,
        expected_name="audio.wav",
    )
    opened = []

    def fake_open_best(path, mode):
        opened.append((path, mode))
        return BytesIO(tar_path.read_bytes())

    close_all()
    monkeypatch.setattr(lazy_pointer, "open_best", fake_open_best)
    monkeypatch.setattr(
        "lhotse.ais.AISRangeReader",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("non-AIStore URLs must retain their existing backend")
        ),
    )

    assert lazy_pointer.read_payload(pointer) == payload
    assert opened == [("https://example.com/audio.tar", "rb")]


def test_ais_byte_range_path_when_sdk_supports_it():
    """When supported, Shar pointers are added as byte-range MOSS requests."""
    pytest.importorskip("aistore")
    from lhotse import AudioSource, Recording
    from lhotse.ais.batch_loader import AISBatchLoader

    rec = Recording(
        id="x",
        sources=[
            AudioSource(
                type="shar_ptr",
                channels=[0],
                source=encode_pointer("ais://b/recording.000000.tar", 1024, 8192),
            )
        ],
        sampling_rate=16000,
        num_samples=16000,
        duration=1.0,
    )

    class FakeBatch:
        def __init__(self):
            self.requests_list = []

    with patch.object(
        AISBatchLoader, "_aistore_byte_range_supported", staticmethod(lambda: True)
    ):
        loader = AISBatchLoader.__new__(AISBatchLoader)
        batch = FakeBatch()
        result = loader._collect_manifest_urls(
            rec,
            batch,
            shar_ptr_uses_batch=True,
            shar_ptr_fallback=[],
            manifest_idx=0,
        )

    assert result is True
    assert len(batch.requests_list) == 1
    moss_in = batch.requests_list[0]
    assert moss_in.obj_name == "recording.000000.tar"
    assert moss_in.start == 1024
    assert moss_in.length == 8192 - 1024
    assert moss_in.archpath is None


def test_ais_individual_get_preserves_mossin_byte_range():
    """If ranged MOSS fails, direct fallback must retry the same byte range."""
    pytest.importorskip("aistore")
    from aistore.sdk.batch.types import MossIn

    from lhotse.ais.batch_loader import AISBatchLoader

    captured = {}

    class FakeReader:
        def read_all(self):
            return b"payload"

    class FakeObject:
        def get_reader(self, *, byte_range=None, archive_config=None):
            captured["byte_range"] = byte_range
            captured["archive_config"] = archive_config
            return FakeReader()

    class FakeBucket:
        def object(self, obj_name):
            captured["obj_name"] = obj_name
            return FakeObject()

    class FakeClient:
        def bucket(self, bck_name, provider):
            captured["bck_name"] = bck_name
            captured["provider"] = provider
            return FakeBucket()

    loader = AISBatchLoader.__new__(AISBatchLoader)
    loader._client = FakeClient()
    moss_in = MossIn.model_construct(
        obj_name="recording.000000.tar",
        bck="b",
        provider="ais",
        start=1024,
        length=8192 - 1024,
    )

    assert loader._get_object_from_moss_in(moss_in) == b"payload"
    assert captured == {
        "bck_name": "b",
        "provider": "ais",
        "obj_name": "recording.000000.tar",
        "byte_range": "bytes=1024-8191",
        "archive_config": None,
    }


# ---------------------------------------------------------------------------
# Pickle / state_dict
# ---------------------------------------------------------------------------


def test_lazy_iterator_pickles(shar_dir_numpy):
    it = LazyIndexedSharIterator(in_dir=shar_dir_numpy, lazy=True)
    _ = next(iter(it))  # warm up
    blob = pickle.dumps(it)
    it2 = pickle.loads(blob)
    assert it2._lazy is True
    c = it2[0]
    assert c.recording.sources[0].type == "shar_ptr"


def test_lazy_state_dict_carries_lazy_flag(shar_dir_numpy):
    it = LazyIndexedSharIterator(in_dir=shar_dir_numpy, lazy=True)
    sd = it.state_dict()
    assert sd["lazy"] is True

    # And it round-trips into a freshly constructed iterator.
    it2 = LazyIndexedSharIterator(in_dir=shar_dir_numpy, lazy=False)
    it2.load_state_dict(sd)
    assert it2._lazy is True


# ---------------------------------------------------------------------------
# Sentinel handling: i == N-1 in the last shard
# ---------------------------------------------------------------------------


def test_sentinel_resolves_for_last_sample(shar_dir_numpy):
    it = LazyIndexedSharIterator(in_dir=shar_dir_numpy, lazy=True)
    last = it[len(it) - 1]
    # Should load fine — exercises the sentinel = file-size code path.
    audio = last.load_audio()
    assert audio.size > 0
