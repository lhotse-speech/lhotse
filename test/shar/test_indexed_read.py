"""
Tests for indexed Shar reader/writer integration (Phase 4).
"""
from pathlib import Path

# Reuse the standard shar test fixture which specifies all fields
from test.shar.conftest import cuts  # noqa: F401

import pytest

from lhotse import CutSet
from lhotse.indexing import index_exists
from lhotse.shar.readers.lazy import LazySharIterator
from lhotse.shar.writers.shar import SharWriter
from lhotse.testing.dummies import DummyManifest

ALL_FIELDS = {
    "recording": "wav",
    "features": "lilcom",
    "custom_embedding": "numpy",
    "custom_features": "lilcom",
    "custom_indexes": "numpy",
    "custom_recording": "wav",
}


# ---------------------------------------------------------------------------
# SharWriter: uncompressed + indexed
# ---------------------------------------------------------------------------


def test_shar_writer_uncompressed(tmp_path, cuts):
    """Write uncompressed JSONL + tar with indexes."""
    writer = SharWriter(
        tmp_path,
        fields=ALL_FIELDS,
        shard_size=10,
        compress_jsonl=False,
        create_index=True,
    )
    with writer:
        for c in cuts:
            writer.write(c)

    # Check that .jsonl (not .jsonl.gz) files were created
    jsonl_files = sorted(tmp_path.glob("cuts.*.jsonl"))
    assert len(jsonl_files) == 2  # 20 cuts / 10 per shard = 2 shards
    gz_files = list(tmp_path.glob("cuts.*.jsonl.gz"))
    assert len(gz_files) == 0

    # Check that .idx files were created for JSONL
    for jf in jsonl_files:
        assert index_exists(jf), f"Missing index for {jf}"

    # Check that .idx files were created for tar
    tar_files = sorted(tmp_path.glob("recording.*.tar"))
    assert len(tar_files) == 2
    for tf in tar_files:
        assert index_exists(tf), f"Missing index for {tf}"


def test_shar_writer_compressed_no_index(tmp_path, cuts):
    """Compressed JSONL doesn't get indexed."""
    writer = SharWriter(
        tmp_path,
        fields=ALL_FIELDS,
        shard_size=10,
        compress_jsonl=True,
        create_index=True,  # index creation skips compressed files
    )
    with writer:
        for c in cuts:
            writer.write(c)

    gz_files = sorted(tmp_path.glob("cuts.*.jsonl.gz"))
    assert len(gz_files) == 2
    # .gz files should NOT have .idx (they're compressed)
    for gf in gz_files:
        assert not index_exists(gf)

    # tar files should still have .idx
    tar_files = sorted(tmp_path.glob("recording.*.tar"))
    for tf in tar_files:
        assert index_exists(tf)


def test_shar_writer_no_index(tmp_path, cuts):
    """create_index=False skips index creation."""
    writer = SharWriter(
        tmp_path,
        fields=ALL_FIELDS,
        shard_size=10,
        compress_jsonl=False,
        create_index=False,
    )
    with writer:
        for c in cuts:
            writer.write(c)

    jsonl_files = sorted(tmp_path.glob("cuts.*.jsonl"))
    assert len(jsonl_files) == 2
    for jf in jsonl_files:
        assert not index_exists(jf)

    tar_files = sorted(tmp_path.glob("recording.*.tar"))
    for tf in tar_files:
        assert not index_exists(tf)


# ---------------------------------------------------------------------------
# LazySharIterator: reading indexed data
# ---------------------------------------------------------------------------


def test_indexed_shar_matches_sequential(tmp_path, cuts):
    """Random access via indexed shar matches sequential iteration."""
    writer = SharWriter(
        tmp_path,
        fields=ALL_FIELDS,
        shard_size=10,
        compress_jsonl=False,
        create_index=True,
    )
    with writer:
        for c in cuts:
            writer.write(c)

    # Read all cuts
    shar_cuts = list(LazySharIterator(in_dir=tmp_path))
    assert len(shar_cuts) == 20

    # Verify IDs match original cuts
    original_ids = sorted(c.id for c in cuts)
    shar_ids = sorted(c.id for c in shar_cuts)
    assert original_ids == shar_ids


def test_shar_auto_detect_indexes(tmp_path, cuts):
    """LazySharIterator works with auto-detected indexes."""
    writer = SharWriter(
        tmp_path,
        fields=ALL_FIELDS,
        shard_size=10,
        compress_jsonl=False,
        create_index=True,
    )
    with writer:
        for c in cuts:
            writer.write(c)

    # .idx files are present, LazySharIterator should read correctly
    shar_iter = LazySharIterator(in_dir=tmp_path)
    shar_cuts = list(shar_iter)
    assert len(shar_cuts) == 20

    # .idx files should not interfere with field detection
    assert "cuts" not in shar_iter.fields  # 'cuts' is always removed from fields
    assert "recording" in shar_iter.fields


# ---------------------------------------------------------------------------
# LazySharIterator: state_dict / load_state_dict
# ---------------------------------------------------------------------------


def test_shar_state_dict_restore(tmp_path, cuts):
    """Checkpoint/restore: first_k + remaining == all_items."""
    writer = SharWriter(
        tmp_path,
        fields=ALL_FIELDS,
        shard_size=10,
        compress_jsonl=False,
        create_index=True,
    )
    with writer:
        for c in cuts:
            writer.write(c)

    # Full uninterrupted run
    all_items = [c.id for c in LazySharIterator(in_dir=tmp_path)]

    # Interrupted run: consume 5 items (within first shard)
    it1 = LazySharIterator(in_dir=tmp_path)
    gen1 = iter(it1)
    first_k = [next(gen1).id for _ in range(5)]
    sd = it1.state_dict()

    # Restored run
    it2 = LazySharIterator(in_dir=tmp_path)
    it2.load_state_dict(sd)
    remaining = [c.id for c in it2]

    assert first_k + remaining == all_items


def test_shar_state_dict_restore_cross_shard(tmp_path, cuts):
    """Checkpoint/restore across a shard boundary (item 15 of 20, shard_size=10)."""
    writer = SharWriter(
        tmp_path,
        fields=ALL_FIELDS,
        shard_size=10,
        compress_jsonl=False,
        create_index=True,
    )
    with writer:
        for c in cuts:
            writer.write(c)

    # Full uninterrupted run
    all_items = [c.id for c in LazySharIterator(in_dir=tmp_path)]
    assert len(all_items) == 20

    # Interrupted run: consume 15 items (crosses shard boundary at 10)
    it1 = LazySharIterator(in_dir=tmp_path)
    gen1 = iter(it1)
    first_k = [next(gen1).id for _ in range(15)]
    sd = it1.state_dict()

    assert sd["current_shard_idx"] == 1  # second shard
    assert sd["position_in_shard"] == 5  # 5 into second shard

    # Restored run
    it2 = LazySharIterator(in_dir=tmp_path)
    it2.load_state_dict(sd)
    remaining = [c.id for c in it2]

    assert first_k + remaining == all_items
