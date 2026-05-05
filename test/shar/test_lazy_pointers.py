"""
Tests for the Lhotse Shar lazy-pointer mode (``LazyIndexedSharIterator(lazy=True)``)
and the supporting ``shar_ptr`` / ``shar_ptr_array`` storage types.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from unittest.mock import patch

# Reuse the standard shar test fixture which specifies all fields
from test.shar.conftest import cuts  # noqa: F401

import numpy as np
import pytest

from lhotse.shar.lazy_pointer import (
    close_all,
    decode_pointer,
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


def test_is_shar_pointer():
    assert is_shar_pointer("/x.tar?o=0&e=10")
    assert not is_shar_pointer("/x.tar")
    assert not is_shar_pointer("/x.tar?o=foo&e=10")
    assert not is_shar_pointer(b"/x.tar?o=0&e=10")  # not a str


def test_decode_pointer_rejects_malformed():
    for bad in ("garbage", "/x.tar?o=10", "/x.tar?o=10&e=20&extra=1"):
        with pytest.raises(ValueError):
            decode_pointer(bad)


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


def test_ais_byte_range_disabled_today():
    """With the currently-installed aistore SDK, byte-range batch is unsupported."""
    pytest.importorskip("aistore")
    from lhotse.ais.batch_loader import AISBatchLoader

    assert AISBatchLoader._aistore_byte_range_supported() is False


def test_ais_collect_returns_false_for_shar_ptr_when_unsupported():
    """When the SDK can't do byte ranges, collect_manifest_urls returns False
    so the caller falls back to the per-cut _prepare_for_reading path."""
    pytest.importorskip("aistore")
    from lhotse.ais.batch_loader import AISBatchLoader
    from lhotse import AudioSource, Recording

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
        loader.client = None  # not touched on this code path
        batch = []
        result = loader._collect_manifest_urls(rec, batch)

    assert result is False
    assert batch == []


def test_ais_byte_range_path_when_sdk_supports_it():
    """Future-proof: when the SDK exposes byte-range batch, the loader routes
    Shar pointers through ``batch.add(start=, length=)``."""
    pytest.importorskip("aistore")
    from lhotse.ais.batch_loader import AISBatchLoader
    from lhotse import AudioSource, Recording

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

    captured = []

    class FakeBatch:
        def add(self, obj, *, start=None, length=None, archpath=None):
            captured.append((obj, start, length, archpath))

    class FakeObject:
        def __init__(self, name):
            self.name = name

    class FakeBucket:
        def object(self, obj_name):
            return FakeObject(obj_name)

    class FakeClient:
        def bucket(self, bck_name, provider):
            return FakeBucket()

    with patch.object(
        AISBatchLoader, "_aistore_byte_range_supported", staticmethod(lambda: True)
    ):
        loader = AISBatchLoader.__new__(AISBatchLoader)
        loader.client = FakeClient()
        batch = FakeBatch()
        result = loader._collect_manifest_urls(rec, batch)

    assert result is True
    assert len(captured) == 1
    obj, start, length, archpath = captured[0]
    assert obj.name == "recording.000000.tar"
    assert start == 1024
    assert length == 8192 - 1024
    assert archpath is None


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
