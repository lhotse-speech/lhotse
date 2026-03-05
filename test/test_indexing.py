import json
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from lhotse.indexing import (
    IndexedJsonlReader,
    IndexedTarReader,
    LazyShuffledRange,
    create_jsonl_index,
    create_shar_index,
    create_tar_index,
    index_exists,
    index_file_path,
    read_index,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def jsonl_file(tmp_path):
    """Create a small uncompressed JSONL file with 10 lines."""
    p = tmp_path / "data.jsonl"
    records = [{"id": f"item-{i}", "value": i * 10} for i in range(10)]
    with open(p, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return p, records


@pytest.fixture
def tar_file(tmp_path):
    """
    Create a small uncompressed Shar-style tar with 5 sample pairs.
    Each sample has a .nodata file and a .nometa file — the simplest Shar pair.
    This avoids the need for valid manifest deserialization in indexing tests.
    """
    import io

    p = tmp_path / "data.tar"
    samples = []
    with tarfile.open(str(p), "w:") as tf:
        for i in range(5):
            # Data member (.nodata = no binary payload)
            data_name = f"sample-{i}.nodata"
            data_info = tarfile.TarInfo(name=data_name)
            data_info.size = 0
            tf.addfile(data_info, io.BytesIO(b""))

            # Metadata member (.nometa = no manifest)
            meta_name = f"sample-{i}.nometa"
            meta_info = tarfile.TarInfo(name=meta_name)
            meta_info.size = 0
            tf.addfile(meta_info, io.BytesIO(b""))

            samples.append((data_name, meta_name))
    return p, samples


# ---------------------------------------------------------------------------
# index_file_path / index_exists
# ---------------------------------------------------------------------------


def test_index_file_path():
    assert index_file_path("cuts.000000.jsonl") == Path("cuts.000000.jsonl.idx")
    assert index_file_path(Path("/a/b/c.tar")) == Path("/a/b/c.tar.idx")


def test_index_exists(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_text("{}\n")
    assert not index_exists(p)
    create_jsonl_index(p)
    assert index_exists(p)


# ---------------------------------------------------------------------------
# JSONL index
# ---------------------------------------------------------------------------


def test_create_jsonl_index(jsonl_file):
    p, records = jsonl_file
    idx_path = create_jsonl_index(p)
    assert idx_path == index_file_path(p)
    assert idx_path.is_file()

    offsets = read_index(idx_path)
    # N records → N+1 entries (last is sentinel)
    assert len(offsets) == len(records) + 1
    # First offset is 0
    assert offsets[0] == 0
    # Sentinel equals file size
    assert offsets[-1] == p.stat().st_size


def test_create_jsonl_index_rejects_compressed(tmp_path):
    p = tmp_path / "data.jsonl.gz"
    p.write_bytes(b"")
    with pytest.raises(RuntimeError, match="compressed"):
        create_jsonl_index(p)


def test_indexed_jsonl_reader(jsonl_file):
    p, records = jsonl_file
    reader = IndexedJsonlReader(p)
    assert len(reader) == len(records)

    # Random access matches expected records
    for i, expected in enumerate(records):
        assert reader[i] == expected

    # Negative indexing
    assert reader[-1] == records[-1]

    # Sequential iteration matches
    assert list(reader) == records


def test_indexed_jsonl_reader_out_of_range(jsonl_file):
    p, records = jsonl_file
    reader = IndexedJsonlReader(p)
    with pytest.raises(IndexError):
        reader[len(records)]
    with pytest.raises(IndexError):
        reader[-(len(records) + 1)]


# ---------------------------------------------------------------------------
# Tar index
# ---------------------------------------------------------------------------


def test_create_tar_index(tar_file):
    p, samples = tar_file
    idx_path = create_tar_index(p)
    assert idx_path == index_file_path(p)
    assert idx_path.is_file()

    offsets = read_index(idx_path)
    # 5 sample pairs → 6 entries (5 + sentinel)
    assert len(offsets) == len(samples) + 1
    # First offset is 0 (first tar header at the start)
    assert offsets[0] == 0
    # Sentinel equals file size
    assert offsets[-1] == p.stat().st_size


def test_create_tar_index_rejects_compressed(tmp_path):
    p = tmp_path / "data.tar.gz"
    p.write_bytes(b"")
    with pytest.raises(RuntimeError, match="compressed"):
        create_tar_index(p)


def test_indexed_tar_reader(tar_file):
    """Random access via IndexedTarReader matches expected pairs."""
    p, samples = tar_file
    reader = IndexedTarReader(p)
    assert len(reader) == len(samples)

    for i in range(len(samples)):
        manifest, data_path = reader[i]
        expected_data_name = samples[i][0]  # nodata filename
        assert str(data_path) == expected_data_name
        assert manifest is None  # .nometa → None manifest


def test_indexed_tar_reader_sequential(tar_file):
    """Iterate over all samples sequentially."""
    p, samples = tar_file
    reader = IndexedTarReader(p)
    results = list(reader)
    assert len(results) == len(samples)
    for (manifest, data_path), sample in zip(results, samples):
        assert str(data_path) == sample[0]
        assert manifest is None


# ---------------------------------------------------------------------------
# Shar index
# ---------------------------------------------------------------------------


def test_create_shar_index(tmp_path, jsonl_file, tar_file):
    # Copy files into a "shar" directory
    shar_dir = tmp_path / "shar"
    shar_dir.mkdir()

    jsonl_p, _ = jsonl_file
    tar_p, _ = tar_file
    import shutil

    shutil.copy(jsonl_p, shar_dir / "cuts.000000.jsonl")
    shutil.copy(tar_p, shar_dir / "recording.000000.tar")
    # Also add a .gz file which should be skipped
    (shar_dir / "cuts.000001.jsonl.gz").write_bytes(b"")

    create_shar_index(shar_dir)

    assert (shar_dir / "cuts.000000.jsonl.idx").is_file()
    assert (shar_dir / "recording.000000.tar.idx").is_file()
    assert not (shar_dir / "cuts.000001.jsonl.gz.idx").is_file()


# ---------------------------------------------------------------------------
# LazyShuffledRange
# ---------------------------------------------------------------------------


def test_lazy_shuffled_range_is_permutation():
    """Every element of range(N) is produced exactly once."""
    for n in [1, 2, 7, 100, 256, 1000]:
        perm = LazyShuffledRange(n, seed=42)
        elements = list(perm)
        assert sorted(elements) == list(range(n)), f"Failed for n={n}"


def test_lazy_shuffled_range_deterministic():
    """Same (N, seed) produces the same permutation."""
    a = list(LazyShuffledRange(500, seed=123))
    b = list(LazyShuffledRange(500, seed=123))
    assert a == b


def test_lazy_shuffled_range_different_seeds():
    """Different seeds produce different permutations."""
    a = list(LazyShuffledRange(500, seed=1))
    b = list(LazyShuffledRange(500, seed=2))
    # They should be valid permutations but differ in order
    assert sorted(a) == sorted(b) == list(range(500))
    assert a != b


def test_lazy_shuffled_range_state_dict():
    """Interrupt mid-iteration, restore, get the same remaining elements."""
    n = 200
    perm = LazyShuffledRange(n, seed=99)

    # Consume first 73 elements
    first_part = [next(perm) for _ in range(73)]
    sd = perm.state_dict()

    # Continue to get remaining elements
    remaining = list(perm)

    # Restore a fresh permutation to the saved position.
    perm2 = LazyShuffledRange(n, seed=99)
    perm2.load_state_dict(sd)
    remaining2 = list(perm2)

    assert remaining == remaining2
    assert sorted(first_part + remaining) == list(range(n))


def test_lazy_shuffled_range_load_state_dict_mismatch():
    """Loading a state_dict with wrong n or seed raises ValueError."""
    perm = LazyShuffledRange(100, seed=42)
    sd = perm.state_dict()

    wrong_n = LazyShuffledRange(200, seed=42)
    with pytest.raises(ValueError, match="state mismatch"):
        wrong_n.load_state_dict(sd)

    wrong_seed = LazyShuffledRange(100, seed=99)
    with pytest.raises(ValueError, match="state mismatch"):
        wrong_seed.load_state_dict(sd)


def test_lazy_shuffled_range_getitem():
    """__getitem__ is consistent with iteration."""
    n = 50
    perm = LazyShuffledRange(n, seed=7)
    indexed = [perm[i] for i in range(n)]
    perm2 = LazyShuffledRange(n, seed=7)
    iterated = list(perm2)
    assert iterated == indexed


def test_lazy_shuffled_range_negative_index():
    perm = LazyShuffledRange(10, seed=0)
    assert perm[-1] == perm[9]
    assert perm[-10] == perm[0]


def test_lazy_shuffled_range_index_error():
    perm = LazyShuffledRange(10, seed=0)
    with pytest.raises(IndexError):
        perm[10]
    with pytest.raises(IndexError):
        perm[-11]


def test_lazy_shuffled_range_len():
    perm = LazyShuffledRange(42, seed=0)
    assert len(perm) == 42


def test_lazy_shuffled_range_edge_cases():
    """Test n=1."""
    perm = LazyShuffledRange(1, seed=0)
    assert list(perm) == [0]


# ---------------------------------------------------------------------------
# Auto-create and missing index
# ---------------------------------------------------------------------------


def test_index_auto_create(jsonl_file):
    """IndexedJsonlReader auto-creates .idx when missing."""
    p, records = jsonl_file
    assert not index_exists(p)
    reader = IndexedJsonlReader(p, auto_create_index=True)
    assert index_exists(p)
    assert len(reader) == len(records)


def test_index_missing_error(jsonl_file):
    """Raises FileNotFoundError when auto_create=False and index is missing."""
    p, _ = jsonl_file
    with pytest.raises(FileNotFoundError):
        IndexedJsonlReader(p, auto_create_index=False)


def test_tar_index_auto_create(tar_file):
    p, samples = tar_file
    assert not index_exists(p)
    reader = IndexedTarReader(p, auto_create_index=True)
    assert index_exists(p)
    assert len(reader) == len(samples)


def test_tar_index_missing_error(tar_file):
    p, _ = tar_file
    with pytest.raises(FileNotFoundError):
        IndexedTarReader(p, auto_create_index=False)


# ---------------------------------------------------------------------------
# LazyIndexedManifestIterator
# ---------------------------------------------------------------------------


def _write_cuts_jsonl(path, n=20):
    """Write a DummyManifest CutSet to an uncompressed JSONL file and return it."""
    from lhotse import CutSet
    from lhotse.testing.dummies import DummyManifest

    cuts = DummyManifest(CutSet, begin_id=0, end_id=n)
    cuts.to_jsonl(path)
    return cuts


def test_lazy_indexed_manifest_iterator_sequential(tmp_path):
    """Sequential (non-shuffled) iteration yields items in file order."""
    from lhotse.lazy import LazyIndexedManifestIterator

    path = tmp_path / "cuts.jsonl"
    original = _write_cuts_jsonl(path)
    original_ids = [c.id for c in original]

    it = LazyIndexedManifestIterator(path, shuffle=False)
    result_ids = [c.id for c in it]

    assert result_ids == original_ids


def test_lazy_indexed_manifest_iterator_shuffled(tmp_path):
    """Shuffled iteration yields all items in a permuted order."""
    from lhotse.lazy import LazyIndexedManifestIterator

    path = tmp_path / "cuts.jsonl"
    original = _write_cuts_jsonl(path)
    original_ids = sorted(c.id for c in original)

    it = LazyIndexedManifestIterator(path, shuffle=True, seed=42)
    result_ids = [c.id for c in it]

    # All items present
    assert sorted(result_ids) == original_ids
    # Order is permuted (with overwhelming probability for n=20)
    assert result_ids != [c.id for c in original]


def test_lazy_indexed_manifest_iterator_getitem(tmp_path):
    """O(1) random access via __getitem__."""
    from lhotse.lazy import LazyIndexedManifestIterator

    path = tmp_path / "cuts.jsonl"
    original = _write_cuts_jsonl(path)
    original_ids = [c.id for c in original]

    it = LazyIndexedManifestIterator(path)

    # Access specific indices
    assert it[0].id == original_ids[0]
    assert it[5].id == original_ids[5]
    assert it[-1].id == original_ids[-1]

    # Out of range
    with pytest.raises(IndexError):
        it[100]


def test_lazy_indexed_manifest_iterator_checkpoint(tmp_path):
    """first_part + remaining == full iteration."""
    from lhotse.lazy import LazyIndexedManifestIterator

    path = tmp_path / "cuts.jsonl"
    _write_cuts_jsonl(path)

    # Full run
    full = [c.id for c in LazyIndexedManifestIterator(path, shuffle=True, seed=42)]

    # Interrupted run: consume first 7
    it1 = LazyIndexedManifestIterator(path, shuffle=True, seed=42)
    gen1 = iter(it1)
    first_k = [next(gen1).id for _ in range(7)]
    sd = it1.state_dict()

    # Restored run
    it2 = LazyIndexedManifestIterator(path, shuffle=True, seed=42)
    it2.load_state_dict(sd)
    remaining = [c.id for c in it2]

    assert first_k + remaining == full


# ---------------------------------------------------------------------------
# has_constant_time_access propagation
# ---------------------------------------------------------------------------


def test_has_constant_time_access_propagates_through_mapper(tmp_path):
    """LazyMapper wrapping an indexed source preserves has_constant_time_access."""
    from lhotse.lazy import LazyIndexedManifestIterator, LazyMapper

    path = tmp_path / "cuts.jsonl"
    _write_cuts_jsonl(path)

    indexed = LazyIndexedManifestIterator(path)
    assert indexed.has_constant_time_access is True

    mapped = LazyMapper(indexed, fn=lambda x: x)
    assert mapped.has_constant_time_access is True

    # __getitem__ should work through the mapper
    direct = indexed[3]
    through_mapper = mapped[3]
    assert direct.id == through_mapper.id


def test_has_constant_time_access_stops_at_filter(tmp_path):
    """LazyFilter always returns has_constant_time_access == False."""
    from lhotse.lazy import LazyFilter, LazyIndexedManifestIterator

    path = tmp_path / "cuts.jsonl"
    _write_cuts_jsonl(path)

    indexed = LazyIndexedManifestIterator(path)
    assert indexed.has_constant_time_access is True

    filtered = LazyFilter(indexed, predicate=lambda x: True)
    assert filtered.has_constant_time_access is False


def test_cutset_from_jsonl_lazy_shuffled(tmp_path):
    """CutSet.from_jsonl_lazy(shuffle=True) yields all items in shuffled order."""
    from lhotse import CutSet

    path = tmp_path / "cuts.jsonl"
    original = _write_cuts_jsonl(path)
    original_ids = sorted(c.id for c in original)

    cs = CutSet.from_jsonl_lazy(path, shuffle=True, seed=42)
    result_ids = [c.id for c in cs]

    assert sorted(result_ids) == original_ids
    assert result_ids != [c.id for c in original]


def test_cutset_from_file_shuffled(tmp_path):
    """CutSet.from_file(shuffle=True) yields all items in deterministic shuffled order."""
    from lhotse import CutSet

    path = tmp_path / "cuts.jsonl"
    original = _write_cuts_jsonl(path)
    original_ids = sorted(c.id for c in original)

    cs = CutSet.from_file(path, shuffle=True, seed=42)
    result_ids = [c.id for c in cs]

    # All items present, order permuted, deterministic
    assert sorted(result_ids) == original_ids
    assert result_ids != [c.id for c in original]
    cs2 = CutSet.from_file(path, shuffle=True, seed=42)
    assert [c.id for c in cs2] == result_ids


def test_cutset_resample_preserves_constant_time_access(tmp_path):
    """CutSet.resample() on an indexed CutSet preserves has_constant_time_access."""
    from lhotse import CutSet

    path = tmp_path / "cuts.jsonl"
    _write_cuts_jsonl(path)

    cs = CutSet.from_file(path, indexed=True)
    assert cs.has_constant_time_access is True

    resampled = cs.resample(24000)
    assert resampled.has_constant_time_access is True


# ---------------------------------------------------------------------------
# is_indexed property propagation
# ---------------------------------------------------------------------------


def test_is_indexed_propagation(tmp_path):
    """is_indexed propagates through the lazy pipeline."""
    from lhotse import CutSet

    path = tmp_path / "cuts.jsonl"
    _write_cuts_jsonl(path)

    cs = CutSet.from_file(path, indexed=True)
    assert cs.is_indexed is True

    # Transforms preserve is_indexed
    cs2 = cs.filter(lambda c: True).repeat(times=2).resample(24000)
    assert cs2.is_indexed is True


# ---------------------------------------------------------------------------
# _origin attachment and propagation
# ---------------------------------------------------------------------------


def test_origin_attachment(tmp_path):
    """_origin is attached to cuts from indexed iterators."""
    from lhotse import CutSet

    path = tmp_path / "cuts.jsonl"
    _write_cuts_jsonl(path)

    cs = CutSet.from_file(path, indexed=True)
    cuts = list(cs)
    for i, c in enumerate(cuts):
        assert hasattr(c, "_origin"), f"Cut {c.id} missing _origin"
        assert c._origin[0] == "lhotse"
        assert c._origin[1] == str(path)
        assert c._origin[2] == i


def test_origin_survives_pipeline(tmp_path):
    """_origin survives filter -> repeat -> mux pipeline."""
    from lhotse import CutSet

    path_a = tmp_path / "cuts_a.jsonl"
    path_b = tmp_path / "cuts_b.jsonl"
    _write_cuts_jsonl(path_a, n=10)
    _write_cuts_jsonl(path_b, n=10)

    even = lambda c: int(c.id.split("-")[-1]) % 2 == 0
    odd = lambda c: int(c.id.split("-")[-1]) % 2 == 1

    a = CutSet.from_file(path_a, indexed=True).filter(even).repeat(times=2)
    b = CutSet.from_file(path_b, indexed=True).filter(odd).repeat(times=2)
    pipeline = CutSet.mux(a, b, weights=[0.5, 0.5], seed=42)

    for c in pipeline:
        assert hasattr(c, "_origin"), f"Cut {c.id} missing _origin after pipeline"
        assert c._origin[0] == "lhotse"


def test_cutset_from_file_indexed(tmp_path):
    """from_file(indexed=True) uses indexed iterator, has_constant_time_access == True."""
    from lhotse import CutSet

    path = tmp_path / "cuts.jsonl"
    _write_cuts_jsonl(path, n=10)

    cs = CutSet.from_file(path, indexed=True)
    assert cs.is_lazy
    assert cs.has_constant_time_access is True
    assert len(list(cs)) == 10


def test_cutset_getitem_indexed(tmp_path):
    """O(1) random access through transform chain on indexed CutSet."""
    from lhotse import CutSet
    from lhotse.lazy import LazyIndexedManifestIterator, LazyMapper

    path = tmp_path / "cuts.jsonl"
    _write_cuts_jsonl(path, n=10)

    indexed = LazyIndexedManifestIterator(path)
    mapped = LazyMapper(indexed, fn=lambda c: c)
    cs = CutSet(mapped)

    assert cs.has_constant_time_access is True

    # Access via CutSet.__getitem__ (int index)
    c0 = cs[0]
    c5 = cs[5]
    assert c0.id == indexed[0].id
    assert c5.id == indexed[5].id


# ---------------------------------------------------------------------------
# CutSet.from_files(indexed=...)
# ---------------------------------------------------------------------------


def test_cutset_from_files_indexed_true(tmp_path):
    """from_files(indexed=True) uses indexed iterators with O(1) access."""
    from lhotse import CutSet

    p1 = tmp_path / "cuts1.jsonl"
    p2 = tmp_path / "cuts2.jsonl"
    _write_cuts_jsonl(p1, n=10)
    _write_cuts_jsonl(p2, n=10)

    cs = CutSet.from_files([p1, p2], shuffle_iters=False, indexed=True)
    assert cs.is_lazy
    assert cs.has_constant_time_access is True
    assert len(list(cs)) == 20


def test_cutset_from_files_indexed_false(tmp_path):
    """from_files(indexed=False) uses streaming iterators, no O(1) access."""
    from lhotse import CutSet

    p1 = tmp_path / "cuts1.jsonl"
    p2 = tmp_path / "cuts2.jsonl"
    _write_cuts_jsonl(p1, n=10)
    _write_cuts_jsonl(p2, n=10)

    cs = CutSet.from_files([p1, p2], shuffle_iters=False, indexed=False)
    assert cs.is_lazy
    assert cs.has_constant_time_access is False
    assert len(list(cs)) == 20


def test_cutset_from_files_indexed_auto_detect(tmp_path):
    """from_files(indexed=None) auto-detects: indexed when .idx exists."""
    from lhotse import CutSet
    from lhotse.indexing import index_exists

    p1 = tmp_path / "cuts1.jsonl"
    p2 = tmp_path / "cuts2.jsonl"
    _write_cuts_jsonl(p1, n=10)
    _write_cuts_jsonl(p2, n=10)

    # No .idx yet — auto-detect should fall back to streaming
    cs_stream = CutSet.from_files([p1, p2], shuffle_iters=False, indexed=None)
    assert cs_stream.has_constant_time_access is False

    # Force-create .idx files by reading in indexed mode once
    CutSet.from_file(p1, indexed=True)
    CutSet.from_file(p2, indexed=True)
    assert index_exists(p1) and index_exists(p2)

    # Now auto-detect should pick up indexed mode
    cs_indexed = CutSet.from_files([p1, p2], shuffle_iters=False, indexed=None)
    assert cs_indexed.has_constant_time_access is True
    assert len(list(cs_indexed)) == 20


def test_cutset_from_files_indexed_getitem(tmp_path):
    """O(1) random access across multiple files via from_files(indexed=True)."""
    from lhotse import CutSet

    p1 = tmp_path / "cuts1.jsonl"
    p2 = tmp_path / "cuts2.jsonl"
    c1 = _write_cuts_jsonl(p1, n=10)
    c2 = _write_cuts_jsonl(p2, n=10)

    cs = CutSet.from_files([p1, p2], shuffle_iters=False, indexed=True)

    # Global index 0 is first item of first file
    assert cs[0].id == list(c1)[0].id
    # Global index 10 is first item of second file
    assert cs[10].id == list(c2)[0].id
    # Negative index
    assert cs[-1].id == list(c2)[-1].id


def test_cutset_from_files_indexed_checkpoint(tmp_path):
    """Checkpoint/restore works with from_files(indexed=True)."""
    from lhotse import CutSet

    p1 = tmp_path / "cuts1.jsonl"
    p2 = tmp_path / "cuts2.jsonl"
    _write_cuts_jsonl(p1, n=10)
    _write_cuts_jsonl(p2, n=10)

    # Full uninterrupted run (no shuffle so order is deterministic)
    all_ids = [
        c.id for c in CutSet.from_files([p1, p2], shuffle_iters=False, indexed=True)
    ]

    # Interrupted at position 5
    cs1 = CutSet.from_files([p1, p2], shuffle_iters=False, indexed=True)
    gen = iter(cs1)
    first_k = [next(gen).id for _ in range(5)]
    sd = cs1.state_dict()

    # Restore
    cs2 = CutSet.from_files([p1, p2], shuffle_iters=False, indexed=True)
    cs2.load_state_dict(sd)
    remaining = [c.id for c in cs2]

    assert first_k + remaining == all_ids


def test_cutset_from_files_indexed_shuffle_across_boundaries(tmp_path):
    """shuffle_iters=True with indexed files shuffles across file boundaries."""
    from lhotse import CutSet

    # Two files with non-overlapping ID ranges
    p1 = tmp_path / "cuts1.jsonl"
    p2 = tmp_path / "cuts2.jsonl"
    _write_cuts_jsonl(p1, n=50)
    # Second file has IDs 50..99
    from lhotse.testing.dummies import DummyManifest

    DummyManifest(CutSet, begin_id=50, end_id=100).to_jsonl(p2)

    cs = CutSet.from_files([p1, p2], shuffle_iters=True, indexed=True, seed=42)
    ids = [c.id for c in cs]
    assert len(ids) == 100
    # All items present
    assert sorted(ids) == sorted([f"dummy-mono-cut-{i:04d}" for i in range(100)])

    # The key property: shuffling crosses file boundaries.
    # Check that items from both files are interleaved, not just
    # "all of file 1 then all of file 2" (or vice versa).
    file1_ids = {f"dummy-mono-cut-{i:04d}" for i in range(50)}
    first_half = ids[:50]
    from_file1_in_first_half = sum(1 for x in first_half if x in file1_ids)
    # With true global shuffle, we expect roughly 25 from each file in
    # the first 50 positions.  If no cross-boundary shuffling happened,
    # we'd see either 0 or 50.
    assert 10 < from_file1_in_first_half < 40, (
        f"Expected interleaving across files, but got "
        f"{from_file1_in_first_half}/50 from file1 in the first half"
    )


def test_cutset_from_files_indexed_shuffle_deterministic(tmp_path):
    """Same seed produces identical order across runs."""
    from lhotse import CutSet

    p1 = tmp_path / "cuts1.jsonl"
    p2 = tmp_path / "cuts2.jsonl"
    _write_cuts_jsonl(p1, n=20)
    from lhotse.testing.dummies import DummyManifest

    DummyManifest(CutSet, begin_id=20, end_id=40).to_jsonl(p2)

    def read():
        return [
            c.id
            for c in CutSet.from_files(
                [p1, p2], shuffle_iters=True, indexed=True, seed=7
            )
        ]

    assert read() == read()


def test_cutset_from_files_indexed_shuffle_all_items_present(tmp_path):
    """Shuffled from_files yields every item exactly once."""
    from lhotse import CutSet

    p1 = tmp_path / "cuts1.jsonl"
    p2 = tmp_path / "cuts2.jsonl"
    _write_cuts_jsonl(p1, n=30)
    from lhotse.testing.dummies import DummyManifest

    DummyManifest(CutSet, begin_id=30, end_id=60).to_jsonl(p2)

    cs = CutSet.from_files([p1, p2], shuffle_iters=True, indexed=True, seed=99)
    ids = [c.id for c in cs]
    assert len(ids) == 60
    assert len(set(ids)) == 60  # no duplicates


def test_cutset_from_files_indexed_shuffle_checkpoint(tmp_path):
    """Checkpoint/restore with globally-shuffled from_files."""
    from lhotse import CutSet

    p1 = tmp_path / "cuts1.jsonl"
    p2 = tmp_path / "cuts2.jsonl"
    _write_cuts_jsonl(p1, n=30)
    from lhotse.testing.dummies import DummyManifest

    DummyManifest(CutSet, begin_id=30, end_id=60).to_jsonl(p2)

    kwargs = dict(paths=[p1, p2], shuffle_iters=True, indexed=True, seed=42)

    # Full uninterrupted run
    all_ids = [c.id for c in CutSet.from_files(**kwargs)]

    # Interrupted at position 15
    cs1 = CutSet.from_files(**kwargs)
    gen = iter(cs1)
    first_k = [next(gen).id for _ in range(15)]
    sd = cs1.state_dict()

    # Restore
    cs2 = CutSet.from_files(**kwargs)
    cs2.load_state_dict(sd)
    remaining = [c.id for c in cs2]

    assert first_k + remaining == all_ids


# ---------------------------------------------------------------------------
# Custom index_path support
# ---------------------------------------------------------------------------


def test_index_exists_with_custom_path(tmp_path, jsonl_file):
    """index_exists checks a custom path instead of the conventional one."""
    p, _ = jsonl_file
    idx_dir = tmp_path / "indexes"
    idx_dir.mkdir()
    custom_idx = idx_dir / "custom.idx"
    assert not index_exists(p, index_path=custom_idx)
    create_jsonl_index(p, output_path=custom_idx)
    assert index_exists(p, index_path=custom_idx)
    # Conventional location should still be empty.
    assert not index_file_path(p).is_file()


def test_create_jsonl_index_output_path(tmp_path, jsonl_file):
    """create_jsonl_index writes to output_path when given."""
    p, records = jsonl_file
    idx_dir = tmp_path / "indexes"
    idx_dir.mkdir()
    custom_idx = idx_dir / "data.jsonl.idx"
    result = create_jsonl_index(p, output_path=custom_idx)
    assert result == custom_idx
    assert custom_idx.is_file()
    # Conventional location should not exist.
    assert not index_file_path(p).is_file()
    offsets = read_index(custom_idx)
    assert len(offsets) == len(records) + 1


def test_create_tar_index_output_path(tmp_path, tar_file):
    """create_tar_index writes to output_path when given."""
    p, samples = tar_file
    idx_dir = tmp_path / "indexes"
    idx_dir.mkdir()
    custom_idx = idx_dir / "data.tar.idx"
    result = create_tar_index(p, output_path=custom_idx)
    assert result == custom_idx
    assert custom_idx.is_file()
    assert not index_file_path(p).is_file()
    offsets = read_index(custom_idx)
    assert len(offsets) == len(samples) + 1


def test_create_shar_index_output_dir(tmp_path, jsonl_file, tar_file):
    """create_shar_index writes .idx files into output_dir."""
    import shutil

    shar_dir = tmp_path / "shar"
    shar_dir.mkdir()
    idx_dir = tmp_path / "indexes"
    idx_dir.mkdir()

    jsonl_p, _ = jsonl_file
    tar_p, _ = tar_file
    shutil.copy(jsonl_p, shar_dir / "cuts.000000.jsonl")
    shutil.copy(tar_p, shar_dir / "recording.000000.tar")

    create_shar_index(shar_dir, output_dir=idx_dir)

    assert (idx_dir / "cuts.000000.jsonl.idx").is_file()
    assert (idx_dir / "recording.000000.tar.idx").is_file()
    # Conventional location next to data should not exist.
    assert not (shar_dir / "cuts.000000.jsonl.idx").is_file()
    assert not (shar_dir / "recording.000000.tar.idx").is_file()


def test_indexed_jsonl_reader_custom_index_path(tmp_path, jsonl_file):
    """IndexedJsonlReader works with a separate .idx location."""
    p, records = jsonl_file
    idx_dir = tmp_path / "indexes"
    idx_dir.mkdir()
    custom_idx = idx_dir / "data.jsonl.idx"
    create_jsonl_index(p, output_path=custom_idx)

    reader = IndexedJsonlReader(p, index_path=custom_idx)
    assert len(reader) == len(records)
    for i, expected in enumerate(records):
        assert reader[i] == expected


def test_indexed_tar_reader_custom_index_path(tmp_path, tar_file):
    """IndexedTarReader works with a separate .idx location."""
    p, samples = tar_file
    idx_dir = tmp_path / "indexes"
    idx_dir.mkdir()
    custom_idx = idx_dir / "data.tar.idx"
    create_tar_index(p, output_path=custom_idx)

    reader = IndexedTarReader(p, index_path=custom_idx)
    assert len(reader) == len(samples)
    for i in range(len(samples)):
        manifest, data_path = reader[i]
        assert str(data_path) == samples[i][0]


def test_indexed_jsonl_reader_custom_index_path_missing_raises(tmp_path, jsonl_file):
    """FileNotFoundError when custom index_path does not exist and auto_create is off."""
    p, _ = jsonl_file
    missing = tmp_path / "indexes" / "nope.idx"
    with pytest.raises(FileNotFoundError):
        IndexedJsonlReader(p, auto_create_index=False, index_path=missing)


def test_indexed_jsonl_reader_custom_index_path_auto_create(tmp_path, jsonl_file):
    """auto_create_index=True creates .idx at the custom index_path location."""
    p, records = jsonl_file
    idx_dir = tmp_path / "indexes"
    idx_dir.mkdir()
    custom_idx = idx_dir / "data.jsonl.idx"

    reader = IndexedJsonlReader(p, auto_create_index=True, index_path=custom_idx)
    assert custom_idx.is_file()
    assert len(reader) == len(records)
    # Conventional location should not exist.
    assert not index_file_path(p).is_file()


def test_lazy_indexed_manifest_iterator_custom_index_path(tmp_path):
    """End-to-end with LazyIndexedManifestIterator and custom index_path."""
    from lhotse.lazy import LazyIndexedManifestIterator

    path = tmp_path / "cuts.jsonl"
    original = _write_cuts_jsonl(path)
    original_ids = [c.id for c in original]

    idx_dir = tmp_path / "indexes"
    idx_dir.mkdir()
    custom_idx = idx_dir / "cuts.jsonl.idx"
    create_jsonl_index(path, output_path=custom_idx)

    it = LazyIndexedManifestIterator(path, index_path=custom_idx)
    result_ids = [c.id for c in it]
    assert result_ids == original_ids


def test_cutset_from_file_with_index_path(tmp_path):
    """CutSet.from_file with custom index_path."""
    from lhotse import CutSet

    path = tmp_path / "cuts.jsonl"
    original = _write_cuts_jsonl(path, n=10)
    original_ids = [c.id for c in original]

    idx_dir = tmp_path / "indexes"
    idx_dir.mkdir()
    custom_idx = idx_dir / "cuts.jsonl.idx"
    create_jsonl_index(path, output_path=custom_idx)

    cs = CutSet.from_file(path, index_path=custom_idx)
    assert cs.is_lazy
    assert cs.has_constant_time_access is True
    result_ids = [c.id for c in cs]
    assert result_ids == original_ids


def test_cutset_from_files_with_index_path(tmp_path):
    """CutSet.from_files with a list of custom indexes."""
    from lhotse import CutSet

    p1 = tmp_path / "cuts1.jsonl"
    p2 = tmp_path / "cuts2.jsonl"
    _write_cuts_jsonl(p1, n=10)
    _write_cuts_jsonl(p2, n=10)

    idx_dir = tmp_path / "indexes"
    idx_dir.mkdir()
    ip1 = idx_dir / "cuts1.jsonl.idx"
    ip2 = idx_dir / "cuts2.jsonl.idx"
    create_jsonl_index(p1, output_path=ip1)
    create_jsonl_index(p2, output_path=ip2)

    cs = CutSet.from_files([p1, p2], shuffle_iters=False, index_path=[ip1, ip2])
    assert cs.has_constant_time_access is True
    assert len(list(cs)) == 20


def test_cutset_from_files_index_path_length_mismatch_raises(tmp_path):
    """ValueError when index_path list length doesn't match paths."""
    from lhotse import CutSet

    p1 = tmp_path / "cuts1.jsonl"
    _write_cuts_jsonl(p1, n=5)

    with pytest.raises(ValueError, match="index_path has"):
        CutSet.from_files([p1], index_path=["a.idx", "b.idx"])
