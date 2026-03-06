"""
Tests for indexed Shar reader/writer integration (Phase 4).
"""
from pathlib import Path

# Reuse the standard shar test fixture which specifies all fields
from test.shar.conftest import cuts  # noqa: F401

import pytest

from lhotse import CutSet
from lhotse.indexing import index_exists
from lhotse.shar.readers.indexed import LazyIndexedSharIterator
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


@pytest.mark.parametrize("output_dir", ["s3://bucket/shar", "pipe:cat > /tmp/shar"])
def test_shar_writer_remote_or_pipe_with_index_enabled_raises(output_dir):
    with pytest.raises(ValueError, match="create_index=False"):
        SharWriter(
            output_dir,
            fields={"recording": "wav"},
            shard_size=10,
            compress_jsonl=False,
            create_index=True,
        )


# ---------------------------------------------------------------------------
# LazySharIterator (streaming): basic reading
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
# LazySharIterator (streaming): state_dict / load_state_dict
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


# ---------------------------------------------------------------------------
# CutSet.to_shar() passthrough
# ---------------------------------------------------------------------------


def test_to_shar_uncompressed_with_index(tmp_path, cuts):
    """CutSet.to_shar(compress_jsonl=False) produces .jsonl + .idx files."""
    cs = CutSet.from_cuts(cuts)
    cs.to_shar(
        output_dir=tmp_path,
        fields=ALL_FIELDS,
        shard_size=10,
        compress_jsonl=False,
        create_index=True,
    )

    # Check that .jsonl (not .jsonl.gz) files were created
    jsonl_files = sorted(tmp_path.glob("cuts.*.jsonl"))
    assert len(jsonl_files) >= 1
    gz_files = list(tmp_path.glob("cuts.*.jsonl.gz"))
    assert len(gz_files) == 0

    # Check that .idx files were created for JSONL
    for jf in jsonl_files:
        assert index_exists(jf), f"Missing index for {jf}"

    # Check that .idx files were created for tar
    tar_files = sorted(tmp_path.glob("recording.*.tar"))
    for tf in tar_files:
        assert index_exists(tf), f"Missing index for {tf}"


def test_cutset_from_shar_with_indexed_data(tmp_path, cuts):
    """CutSet.from_shar(in_dir=...) works correctly with indexed Shar data."""
    cs = CutSet.from_cuts(cuts)
    cs.to_shar(
        output_dir=tmp_path,
        fields=ALL_FIELDS,
        shard_size=10,
        compress_jsonl=False,
        create_index=True,
    )

    # Read back via CutSet.from_shar (the user-facing API)
    shar_cuts = CutSet.from_shar(in_dir=tmp_path)
    shar_ids = sorted(c.id for c in shar_cuts)
    original_ids = sorted(c.id for c in cuts)
    assert shar_ids == original_ids


# ---------------------------------------------------------------------------
# LazyIndexedSharIterator: has_constant_time_access and __getitem__
# ---------------------------------------------------------------------------


def test_shar_has_constant_time_access(tmp_path, cuts):
    """Indexed Shar reports has_constant_time_access=True."""
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

    shar_iter = LazyIndexedSharIterator(in_dir=tmp_path)
    assert shar_iter.has_constant_time_access is True
    assert shar_iter.is_indexed is True


def test_shar_getitem(tmp_path, cuts):
    """__getitem__ returns the same cuts as sequential iteration."""
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

    shar_iter = LazyIndexedSharIterator(in_dir=tmp_path)
    sequential = list(LazyIndexedSharIterator(in_dir=tmp_path))

    # Check specific indices including cross-shard and negative
    for idx in [0, 5, 19, -1]:
        assert shar_iter[idx].id == sequential[idx].id

    # Check all indices match
    for i in range(len(sequential)):
        assert shar_iter[i].id == sequential[i].id


def test_shar_no_index_no_constant_time_access(tmp_path, cuts):
    """LazySharIterator (streaming) always reports is_indexed=False."""
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

    shar_iter = LazySharIterator(in_dir=tmp_path)
    assert shar_iter.is_indexed is False


# ---------------------------------------------------------------------------
# LazyIndexedSharIterator: state_dict / load_state_dict
# ---------------------------------------------------------------------------


def test_indexed_shar_state_dict_restore(tmp_path, cuts):
    """Checkpoint/restore with LazyIndexedSharIterator."""
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
    all_items = [c.id for c in LazyIndexedSharIterator(in_dir=tmp_path)]

    # Interrupted run: consume 5 items
    it1 = LazyIndexedSharIterator(in_dir=tmp_path)
    gen1 = iter(it1)
    first_k = [next(gen1).id for _ in range(5)]
    sd = it1.state_dict()

    # Restored run
    it2 = LazyIndexedSharIterator(in_dir=tmp_path)
    it2.load_state_dict(sd)
    remaining = [c.id for c in it2]

    assert first_k + remaining == all_items


def test_indexed_shar_state_dict_restore_cross_shard(tmp_path, cuts):
    """Checkpoint/restore across a shard boundary with LazyIndexedSharIterator."""
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
    all_items = [c.id for c in LazyIndexedSharIterator(in_dir=tmp_path)]
    assert len(all_items) == 20

    # Interrupted run: consume 15 items (crosses shard boundary at 10)
    it1 = LazyIndexedSharIterator(in_dir=tmp_path)
    gen1 = iter(it1)
    first_k = [next(gen1).id for _ in range(15)]
    sd = it1.state_dict()

    assert sd["position"] == 15

    # Restored run
    it2 = LazyIndexedSharIterator(in_dir=tmp_path)
    it2.load_state_dict(sd)
    remaining = [c.id for c in it2]

    assert first_k + remaining == all_items


# ---------------------------------------------------------------------------
# LazyIndexedSharIterator: shuffle and __len__
# ---------------------------------------------------------------------------


def test_indexed_shar_shuffle(tmp_path, cuts):
    """Shuffled iteration yields all items."""
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

    shar_iter = LazyIndexedSharIterator(in_dir=tmp_path, shuffle=True, seed=42)
    shuffled = list(shar_iter)
    assert len(shuffled) == 20

    # All IDs should be present
    expected_ids = sorted(c.id for c in cuts)
    actual_ids = sorted(c.id for c in shuffled)
    assert actual_ids == expected_ids

    # Order should differ from sequential (with overwhelming probability)
    sequential = list(LazyIndexedSharIterator(in_dir=tmp_path))
    seq_ids = [c.id for c in sequential]
    shuf_ids = [c.id for c in shuffled]
    assert shuf_ids != seq_ids


def test_indexed_shar_len(tmp_path, cuts):
    """__len__ matches item count."""
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

    shar_iter = LazyIndexedSharIterator(in_dir=tmp_path)
    assert len(shar_iter) == 20


# ---------------------------------------------------------------------------
# LazySharIterator: is_indexed always False
# ---------------------------------------------------------------------------


def test_lazy_shar_is_indexed_false_for_gz(tmp_path, cuts):
    """Compressed shards -> LazySharIterator.is_indexed = False."""
    writer = SharWriter(
        tmp_path,
        fields=ALL_FIELDS,
        shard_size=10,
        compress_jsonl=True,
        create_index=True,
    )
    with writer:
        for c in cuts:
            writer.write(c)

    shar_iter = LazySharIterator(in_dir=tmp_path)
    assert shar_iter.is_indexed is False


# ---------------------------------------------------------------------------
# CutSet.from_shar(indexed=...) parameter
# ---------------------------------------------------------------------------


def test_cutset_from_shar_indexed_true(tmp_path, cuts):
    """CutSet.from_shar(indexed=True) uses LazyIndexedSharIterator."""
    cs = CutSet.from_cuts(cuts)
    cs.to_shar(
        output_dir=tmp_path,
        fields=ALL_FIELDS,
        shard_size=10,
        compress_jsonl=False,
        create_index=True,
    )

    shar_cs = CutSet.from_shar(in_dir=tmp_path, indexed=True)
    assert shar_cs.is_indexed is True
    assert len(list(shar_cs)) == 20


def test_cutset_from_shar_indexed_false(tmp_path, cuts):
    """CutSet.from_shar(indexed=False) uses LazySharIterator."""
    cs = CutSet.from_cuts(cuts)
    cs.to_shar(
        output_dir=tmp_path,
        fields=ALL_FIELDS,
        shard_size=10,
        compress_jsonl=False,
        create_index=True,
    )

    shar_cs = CutSet.from_shar(in_dir=tmp_path, indexed=False)
    assert shar_cs.is_indexed is False
    assert len(list(shar_cs)) == 20


def test_cutset_from_shar_indexed_auto_detect(tmp_path, cuts):
    """CutSet.from_shar(indexed=None) auto-detects indexed mode."""
    cs = CutSet.from_cuts(cuts)
    cs.to_shar(
        output_dir=tmp_path,
        fields=ALL_FIELDS,
        shard_size=10,
        compress_jsonl=False,
        create_index=True,
    )

    # Auto-detect: uncompressed + .idx exists -> indexed
    shar_cs = CutSet.from_shar(in_dir=tmp_path)
    assert shar_cs.is_indexed is True


def test_cutset_from_shar_indexed_auto_detect_compressed(tmp_path, cuts):
    """CutSet.from_shar(indexed=None) falls back to streaming for compressed."""
    cs = CutSet.from_cuts(cuts)
    cs.to_shar(
        output_dir=tmp_path,
        fields=ALL_FIELDS,
        shard_size=10,
        compress_jsonl=True,
        create_index=True,
    )

    # Auto-detect: compressed -> streaming
    shar_cs = CutSet.from_shar(in_dir=tmp_path)
    assert shar_cs.is_indexed is False


@pytest.mark.parametrize("slice_length", [0, 5])
def test_cutset_from_shar_indexed_rejects_streaming_params(
    tmp_path, cuts, slice_length
):
    """CutSet.from_shar(indexed=True) rejects streaming-only params."""
    cs = CutSet.from_cuts(cuts)
    cs.to_shar(
        output_dir=tmp_path,
        fields=ALL_FIELDS,
        shard_size=10,
        compress_jsonl=False,
        create_index=True,
    )

    with pytest.raises(ValueError, match="not supported with indexed=True"):
        CutSet.from_shar(in_dir=tmp_path, indexed=True, slice_length=slice_length)


def test_cutset_from_shar_indexed_split_for_dataloading(tmp_path, cuts):
    """CutSet.from_shar(indexed=True, split_for_dataloading=True) works."""
    cs = CutSet.from_cuts(cuts)
    cs.to_shar(
        output_dir=tmp_path,
        fields=ALL_FIELDS,
        shard_size=10,
        compress_jsonl=False,
        create_index=True,
    )

    # Outside a DataLoader, split_for_dataloading is a no-op (1 node, 1 worker).
    shar_cs = CutSet.from_shar(
        in_dir=tmp_path, indexed=True, split_for_dataloading=True
    )
    assert shar_cs.is_indexed is True
    assert len(list(shar_cs)) == 20


# ---------------------------------------------------------------------------
# LazyIndexedSharIterator: fields-based mode
# ---------------------------------------------------------------------------


def test_indexed_shar_fields_based(tmp_path):
    """LazyIndexedSharIterator works with fields-based construction."""
    path = tmp_path / "cuts.000000.jsonl"
    DummyManifest(CutSet, begin_id=0, end_id=5).to_jsonl(path)

    shar = LazyIndexedSharIterator(fields={"cuts": [str(path)]})
    assert shar.is_indexed is True
    assert len(shar) == 5

    cuts = list(shar)
    assert len(cuts) == 5
    for i, c in enumerate(cuts):
        assert hasattr(c, "_graph_origin")
        assert c._graph_origin == (i, 0)


# ---------------------------------------------------------------------------
# LazyIndexedSharIterator: pickling
# ---------------------------------------------------------------------------


def test_indexed_shar_pickle(tmp_path, cuts):
    """LazyIndexedSharIterator survives pickling."""
    import pickle

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

    shar_iter = LazyIndexedSharIterator(in_dir=tmp_path)
    # Access an item to populate caches
    _ = shar_iter[0]

    # Pickle and unpickle
    data = pickle.dumps(shar_iter)
    restored = pickle.loads(data)

    assert len(restored) == 20
    assert restored[0].id == shar_iter[0].id
    assert list(c.id for c in restored) == list(
        c.id for c in LazyIndexedSharIterator(in_dir=tmp_path)
    )


# ---------------------------------------------------------------------------
# LazyIndexedSharIterator: compressed shard rejection
# ---------------------------------------------------------------------------


def test_indexed_shar_rejects_compressed(tmp_path, cuts):
    """LazyIndexedSharIterator raises ValueError for compressed JSONL."""
    writer = SharWriter(
        tmp_path,
        fields=ALL_FIELDS,
        shard_size=10,
        compress_jsonl=True,
        create_index=True,
    )
    with writer:
        for c in cuts:
            writer.write(c)

    with pytest.raises(ValueError, match="uncompressed local"):
        LazyIndexedSharIterator(in_dir=tmp_path)


# ---------------------------------------------------------------------------
# LazyIndexedSharIterator: graph token roundtrip
# ---------------------------------------------------------------------------


def test_indexed_shar_graph_token_roundtrip(tmp_path, cuts):
    """Graph tokens round-trip for in_dir-based LazyIndexedSharIterator."""
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

    shar = LazyIndexedSharIterator(in_dir=tmp_path)
    shar_cuts = list(shar)

    # Check that graph tokens work for a few indices.
    for idx in [0, 5, 19]:
        c = shar_cuts[idx]
        assert c._graph_origin == (idx, 0)
        reloaded = shar[c._graph_origin]
        assert reloaded.id == c.id


# ---------------------------------------------------------------------------
# LazyIndexedSharIterator: out-of-range indexing
# ---------------------------------------------------------------------------


def test_indexed_shar_fields_graph_token_roundtrip(tmp_path, cuts):
    """Graph tokens round-trip for fields-based LazyIndexedSharIterator."""
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

    # Build fields dict manually (same as in_dir discovery but explicit).
    cuts_jsonl = sorted(tmp_path.glob("cuts.*.jsonl"))
    fields = {"cuts": [str(p) for p in cuts_jsonl]}
    for field_name in ALL_FIELDS:
        ext = "tar" if ALL_FIELDS[field_name] != "jsonl" else "jsonl"
        paths = sorted(tmp_path.glob(f"{field_name}.*.{ext}"))
        if paths:
            fields[field_name] = [str(p) for p in paths]

    shar = LazyIndexedSharIterator(fields=fields)
    shar_cuts = list(shar)
    assert len(shar_cuts) == 20

    # Verify graph-token roundtrip for a few indices.
    for idx in [0, 5, 19]:
        c = shar_cuts[idx]
        assert c._graph_origin == (idx, 0)
        reloaded = shar[c._graph_origin]
        assert reloaded.id == c.id
        # Verify field data was reloaded (recording should have audio data attached).
        assert hasattr(reloaded, "recording") and reloaded.recording is not None


def test_indexed_shar_getitem_out_of_range(tmp_path, cuts):
    """__getitem__ with out-of-range index raises IndexError."""
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

    shar_iter = LazyIndexedSharIterator(in_dir=tmp_path)

    with pytest.raises(IndexError):
        shar_iter[20]

    with pytest.raises(IndexError):
        shar_iter[-21]


# ---------------------------------------------------------------------------
# Custom index_path support
# ---------------------------------------------------------------------------


def _write_shar_and_move_indexes(tmp_path, cuts, *, idx_dir_name="indexes"):
    """
    Write shar data, then move all .idx files from the data dir into
    a separate index directory.  Returns ``(data_dir, idx_dir)``.
    """
    import shutil

    data_dir = tmp_path / "data"
    idx_dir = tmp_path / idx_dir_name
    data_dir.mkdir(exist_ok=True)
    idx_dir.mkdir(exist_ok=True)

    writer = SharWriter(
        data_dir,
        fields=ALL_FIELDS,
        shard_size=10,
        compress_jsonl=False,
        create_index=True,
    )
    with writer:
        for c in cuts:
            writer.write(c)

    # Move all .idx files to the index directory.
    for idx_file in sorted(data_dir.glob("*.idx")):
        shutil.move(str(idx_file), str(idx_dir / idx_file.name))

    # Verify conventional .idx files are gone.
    assert list(data_dir.glob("*.idx")) == []
    return data_dir, idx_dir


def test_indexed_shar_in_dir_with_index_path_dir(tmp_path, cuts):
    """Read shar with .idx files stored in a separate directory (in_dir mode)."""
    data_dir, idx_dir = _write_shar_and_move_indexes(tmp_path, cuts)

    shar = LazyIndexedSharIterator(in_dir=data_dir, index_path=idx_dir)
    shar_cuts = list(shar)
    assert len(shar_cuts) == 20

    original_ids = sorted(c.id for c in cuts)
    assert sorted(c.id for c in shar_cuts) == original_ids


def test_indexed_shar_fields_with_index_path_dict(tmp_path, cuts):
    """Read shar with .idx paths supplied via dict (fields mode)."""
    data_dir, idx_dir = _write_shar_and_move_indexes(tmp_path, cuts)

    # Build fields dict from the data directory.
    cuts_jsonl = sorted(data_dir.glob("cuts.*.jsonl"))
    fields = {"cuts": [str(p) for p in cuts_jsonl]}
    index_dict = {"cuts": [str(idx_dir / (p.name + ".idx")) for p in cuts_jsonl]}
    for field_name in ALL_FIELDS:
        tar_paths = sorted(data_dir.glob(f"{field_name}.*.tar"))
        if tar_paths:
            fields[field_name] = [str(p) for p in tar_paths]
            index_dict[field_name] = [
                str(idx_dir / (p.name + ".idx")) for p in tar_paths
            ]

    shar = LazyIndexedSharIterator(fields=fields, index_path=index_dict)
    shar_cuts = list(shar)
    assert len(shar_cuts) == 20

    original_ids = sorted(c.id for c in cuts)
    assert sorted(c.id for c in shar_cuts) == original_ids


def test_cutset_from_shar_in_dir_with_index_path(tmp_path, cuts):
    """End-to-end: CutSet.from_shar(in_dir=..., index_path=...)."""
    data_dir, idx_dir = _write_shar_and_move_indexes(tmp_path, cuts)

    cs = CutSet.from_shar(in_dir=data_dir, index_path=idx_dir)
    assert cs.is_indexed is True
    shar_ids = sorted(c.id for c in cs)
    assert shar_ids == sorted(c.id for c in cuts)


def test_cutset_from_shar_fields_with_index_path(tmp_path, cuts):
    """End-to-end: CutSet.from_shar(fields=..., index_path=...)."""
    data_dir, idx_dir = _write_shar_and_move_indexes(tmp_path, cuts)

    cuts_jsonl = sorted(data_dir.glob("cuts.*.jsonl"))
    fields = {"cuts": [str(p) for p in cuts_jsonl]}
    index_dict = {"cuts": [str(idx_dir / (p.name + ".idx")) for p in cuts_jsonl]}
    for field_name in ALL_FIELDS:
        tar_paths = sorted(data_dir.glob(f"{field_name}.*.tar"))
        if tar_paths:
            fields[field_name] = [str(p) for p in tar_paths]
            index_dict[field_name] = [
                str(idx_dir / (p.name + ".idx")) for p in tar_paths
            ]

    cs = CutSet.from_shar(fields=fields, index_path=index_dict)
    assert cs.is_indexed is True
    assert len(list(cs)) == 20


def test_indexed_shar_index_path_missing_file_raises(tmp_path, cuts):
    """Missing .idx in index dir triggers auto-create or error."""
    data_dir, idx_dir = _write_shar_and_move_indexes(tmp_path, cuts)
    # Delete one .idx file to simulate a missing index.
    idx_files = sorted(idx_dir.glob("cuts.*.idx"))
    assert len(idx_files) > 0
    idx_files[0].unlink()

    # With auto_create=True (default), it should recreate the missing .idx
    # at the custom location.
    shar = LazyIndexedSharIterator(in_dir=data_dir, index_path=idx_dir)
    assert len(shar) == 20


def test_indexed_shar_index_path_dict_key_mismatch_raises(tmp_path, cuts):
    """Dict key mismatch raises ValueError."""
    data_dir, _ = _write_shar_and_move_indexes(tmp_path, cuts)

    cuts_jsonl = sorted(data_dir.glob("cuts.*.jsonl"))
    fields = {"cuts": [str(p) for p in cuts_jsonl]}

    with pytest.raises(ValueError, match="does not match any field"):
        LazyIndexedSharIterator(
            fields=fields,
            index_path={"nonexistent_field": ["foo.idx"]},
        )


def test_indexed_shar_pickle_with_index_path(tmp_path, cuts):
    """Pickling preserves index_path."""
    import pickle

    data_dir, idx_dir = _write_shar_and_move_indexes(tmp_path, cuts)

    shar = LazyIndexedSharIterator(in_dir=data_dir, index_path=idx_dir)
    _ = shar[0]  # populate caches

    data = pickle.dumps(shar)
    restored = pickle.loads(data)

    assert len(restored) == 20
    assert restored[0].id == shar[0].id
    # Verify all items accessible after unpickling.
    assert len(list(restored)) == 20


def test_cutset_from_shar_index_path_with_indexed_false_raises(tmp_path, cuts):
    """Setting index_path with indexed=False raises ValueError."""
    data_dir, idx_dir = _write_shar_and_move_indexes(tmp_path, cuts)

    with pytest.raises(ValueError, match="contradictory"):
        CutSet.from_shar(in_dir=data_dir, indexed=False, index_path=idx_dir)
