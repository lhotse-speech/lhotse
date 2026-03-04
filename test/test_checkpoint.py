"""
Tests for lhotse/checkpoint.py — iterator graph traversal and DataloaderCheckpoint.

Every collect/restore test verifies the core property:
    items_before_checkpoint + items_after_restore == all_items_uninterrupted
"""
import json

import pytest

from lhotse.checkpoint import (
    DataloaderCheckpoint,
    collect_state_dict,
    restore_state_dict,
)
from lhotse.lazy import (
    LazyFilter,
    LazyIteratorChain,
    LazyIteratorMultiplexer,
    LazyJsonlIterator,
    LazyMapper,
    LazyRepeater,
    LazyShuffler,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_jsonl(tmp_path, n=10, name="data.jsonl"):
    from lhotse.indexing import create_jsonl_index

    p = tmp_path / name
    with open(p, "w") as f:
        for i in range(n):
            f.write(json.dumps({"id": f"item-{i}", "value": i}) + "\n")
    create_jsonl_index(p)
    return p


def _consume(it, k):
    return [next(it) for _ in range(k)]


# ---------------------------------------------------------------------------
# collect / restore — single iterator
# ---------------------------------------------------------------------------


def test_collect_restore_simple(tmp_path):
    """Single LazyJsonlIterator: interrupted + resumed == full run."""
    p = _make_jsonl(tmp_path, n=10)

    all_items = list(LazyJsonlIterator(p))

    # Interrupted: consume 5, checkpoint via collect_state_dict
    it1 = LazyJsonlIterator(p)
    gen1 = iter(it1)
    first_k = _consume(gen1, 5)
    sd = collect_state_dict(it1)

    # Restore via restore_state_dict
    it2 = LazyJsonlIterator(p)
    restore_state_dict(it2, sd)
    remaining = list(it2)

    assert first_k + remaining == all_items


# ---------------------------------------------------------------------------
# collect / restore — nested stateful pipeline
# ---------------------------------------------------------------------------


def test_collect_restore_nested(tmp_path):
    """Multi-level: Repeater(Filter(Chain(A, B))) — all nodes stateful."""
    p1 = _make_jsonl(tmp_path, n=5, name="a.jsonl")
    p2 = _make_jsonl(tmp_path, n=5, name="b.jsonl")

    def make_pipeline():
        s1 = LazyJsonlIterator(p1)
        s2 = LazyJsonlIterator(p2)
        chain = LazyIteratorChain(s1, s2)
        filt = LazyFilter(chain, predicate=lambda x: True)
        rep = LazyRepeater(filt, times=2)
        return rep

    # Full run: 10 items × 2 epochs = 20
    all_items = list(make_pipeline())
    assert len(all_items) == 20

    # Interrupted: consume 7 items (mid-way through epoch 0)
    pipe1 = make_pipeline()
    gen1 = iter(pipe1)
    first_k = _consume(gen1, 7)
    sd = collect_state_dict(pipe1)

    # Restore
    pipe2 = make_pipeline()
    restore_state_dict(pipe2, sd)
    remaining = list(pipe2)

    assert first_k + remaining == all_items


# ---------------------------------------------------------------------------
# collect / restore — Filter + Mapper
# ---------------------------------------------------------------------------


def test_collect_restore_with_filter_mapper(tmp_path):
    """Filter and Mapper pipeline: interrupted + resumed == full run."""
    p = _make_jsonl(tmp_path, n=10)

    def make_pipeline():
        inner = LazyJsonlIterator(p)
        filt = LazyFilter(inner, predicate=lambda x: x["value"] % 2 == 0)
        mapper = LazyMapper(filt, fn=lambda x: {**x, "doubled": x["value"] * 2})
        return mapper

    # Full run: 5 even values (0, 2, 4, 6, 8)
    all_items = list(make_pipeline())
    assert len(all_items) == 5

    # Interrupted: consume 3
    pipe1 = make_pipeline()
    gen1 = iter(pipe1)
    first_k = _consume(gen1, 3)
    sd = collect_state_dict(pipe1)

    # Restore
    pipe2 = make_pipeline()
    restore_state_dict(pipe2, sd)
    remaining = list(pipe2)

    assert first_k + remaining == all_items


# ---------------------------------------------------------------------------
# Non-checkpointable node detection (LazyShuffler)
# ---------------------------------------------------------------------------


def test_collect_raises_on_non_checkpointable_node(tmp_path):
    """
    collect_state_dict raises NotImplementedError when the pipeline contains
    a non-checkpointable node like LazyShuffler, giving the user a clear
    message rather than silently producing an incomplete checkpoint.
    """
    p1 = _make_jsonl(tmp_path, n=5, name="a.jsonl")
    p2 = _make_jsonl(tmp_path, n=5, name="b.jsonl")

    s1 = LazyJsonlIterator(p1)
    s2 = LazyJsonlIterator(p2)
    chain = LazyIteratorChain(s1, s2)
    shuf = LazyShuffler(chain, buffer_size=3)
    rep = LazyRepeater(shuf, times=2)

    with pytest.raises(
        NotImplementedError, match="LazyShuffler does not support checkpointing"
    ):
        collect_state_dict(rep)


# ---------------------------------------------------------------------------
# Type mismatch error
# ---------------------------------------------------------------------------


def test_type_mismatch_error(tmp_path):
    """restore_state_dict raises TypeError on node type mismatch."""
    p = _make_jsonl(tmp_path)
    it = LazyJsonlIterator(p)
    filt = LazyFilter([1, 2, 3], predicate=lambda x: True)

    sd = collect_state_dict(it)
    with pytest.raises(TypeError, match="Type mismatch"):
        restore_state_dict(filt, sd)


# ---------------------------------------------------------------------------
# DataloaderCheckpoint save / load
# ---------------------------------------------------------------------------


def test_dataloader_checkpoint_save_load(tmp_path):
    """File I/O round-trip."""
    ckpt = DataloaderCheckpoint(
        num_workers=2,
        world_size=4,
        rank=1,
        worker_states=[{"a": 1}, {"b": 2}],
        sampler_state={"epoch": 3, "diagnostics": {}},
    )
    path = tmp_path / "ckpt.json"
    ckpt.save(path)

    loaded = DataloaderCheckpoint.load(path)
    assert loaded.num_workers == 2
    assert loaded.world_size == 4
    assert loaded.rank == 1
    assert loaded.worker_states == [{"a": 1}, {"b": 2}]
    assert loaded.sampler_state["epoch"] == 3


def test_dataloader_checkpoint_save_load_with_rng_state(tmp_path):
    """Round-trip with RNG state (tuples get serialized as lists)."""
    import random

    rng = random.Random(42)
    rng_state = rng.getstate()

    ckpt = DataloaderCheckpoint(
        num_workers=1,
        world_size=1,
        rank=0,
        worker_states=[{"rng_state": rng_state}],
        sampler_state={},
    )
    path = tmp_path / "ckpt_rng.json"
    ckpt.save(path)

    loaded = DataloaderCheckpoint.load(path)
    # Tuples become lists in JSON
    assert loaded.worker_states[0]["rng_state"] is not None


# ---------------------------------------------------------------------------
# Checkpoint validation
# ---------------------------------------------------------------------------


def test_checkpoint_validation_mismatch(tmp_path):
    """Error on num_workers/world_size/rank mismatch."""
    ckpt = DataloaderCheckpoint(
        num_workers=2,
        world_size=4,
        rank=0,
    )
    with pytest.raises(ValueError, match="num_workers"):
        ckpt.validate(num_workers=3, world_size=4, rank=0)
    with pytest.raises(ValueError, match="world_size"):
        ckpt.validate(num_workers=2, world_size=8, rank=0)
    with pytest.raises(ValueError, match="rank"):
        ckpt.validate(num_workers=2, world_size=4, rank=1)
    # No error when matching
    ckpt.validate(num_workers=2, world_size=4, rank=0)


# ---------------------------------------------------------------------------
# Complex multiplexer integration test
# ---------------------------------------------------------------------------


def test_collect_restore_multiplexed_filtered_repeated(tmp_path):
    """
    Complex pipeline: 3 sources of different lengths, each filtered (even only)
    and repeated different times, fed into a weighted multiplexer.

    Pipeline:
        A(5) --filter(even)--> 3 items --repeat(4)--> 12
        B(11) --filter(even)--> 6 items --repeat(3)--> 18
        C(19) --filter(even)--> 10 items --repeat(2)--> 20
        Multiplexer(A, B, C, weights=[0.1, 0.6, 0.3], seed=42) --> 50 items
    """
    pa = _make_jsonl(tmp_path, n=5, name="a.jsonl")
    pb = _make_jsonl(tmp_path, n=11, name="b.jsonl")
    pc = _make_jsonl(tmp_path, n=19, name="c.jsonl")

    is_even = lambda x: x["value"] % 2 == 0

    def make_pipeline():
        a = LazyRepeater(LazyFilter(LazyJsonlIterator(pa), predicate=is_even), times=4)
        b = LazyRepeater(LazyFilter(LazyJsonlIterator(pb), predicate=is_even), times=3)
        c = LazyRepeater(LazyFilter(LazyJsonlIterator(pc), predicate=is_even), times=2)
        return LazyIteratorMultiplexer(a, b, c, weights=[0.1, 0.6, 0.3], seed=42)

    # Full run
    all_items = list(make_pipeline())
    assert len(all_items) == 50

    # Interrupted: consume 20 items, checkpoint
    pipe1 = make_pipeline()
    gen1 = iter(pipe1)
    first_k = _consume(gen1, 20)
    sd = collect_state_dict(pipe1)

    # Restore
    pipe2 = make_pipeline()
    restore_state_dict(pipe2, sd)
    remaining = list(pipe2)

    assert first_k + remaining == all_items


# ---------------------------------------------------------------------------
# Origin registry: reload_from_origin
# ---------------------------------------------------------------------------


def test_origin_roundtrip(tmp_path):
    """reload_from_origin correctly re-reads a cut."""
    from lhotse import CutSet
    from lhotse.checkpoint import reload_from_origin
    from lhotse.testing.dummies import DummyManifest

    path = tmp_path / "cuts.jsonl"
    DummyManifest(CutSet, begin_id=0, end_id=5).to_jsonl(path)

    cs = CutSet.from_file(path, indexed=True)
    cuts = list(cs)
    for c in cuts:
        reloaded = reload_from_origin(c._origin)
        assert reloaded.id == c.id
        assert reloaded.duration == c.duration


# ---------------------------------------------------------------------------
# CutSet state_dict / load_state_dict
# ---------------------------------------------------------------------------


def test_cutset_state_dict_basic(tmp_path):
    """Lazy CutSet round-trip: first_k + remaining == full."""
    from lhotse import CutSet
    from lhotse.lazy import LazyIndexedManifestIterator
    from lhotse.testing.dummies import DummyManifest

    path = tmp_path / "cuts.jsonl"
    DummyManifest(CutSet, begin_id=0, end_id=20).to_jsonl(path)

    # Full run
    full_ids = [c.id for c in CutSet(LazyIndexedManifestIterator(path))]

    # Interrupted run
    cs1 = CutSet(LazyIndexedManifestIterator(path))
    gen1 = iter(cs1)
    first_k = [next(gen1).id for _ in range(8)]
    sd = cs1.state_dict()

    # Restored run
    cs2 = CutSet(LazyIndexedManifestIterator(path))
    cs2.load_state_dict(sd)
    remaining = [c.id for c in cs2]

    assert first_k + remaining == full_ids


def test_cutset_state_dict_with_transforms(tmp_path):
    """CutSet state_dict through resample (map transform)."""
    from lhotse import CutSet
    from lhotse.testing.dummies import DummyManifest

    path = tmp_path / "cuts.jsonl"
    DummyManifest(CutSet, begin_id=0, end_id=20).to_jsonl(path)

    def make():
        return CutSet.from_file(path, indexed=True).resample(24000)

    full_ids = [c.id for c in make()]

    cs1 = make()
    gen1 = iter(cs1)
    first_k = [next(gen1).id for _ in range(5)]
    sd = cs1.state_dict()

    cs2 = make()
    cs2.load_state_dict(sd)
    remaining = [c.id for c in cs2]

    assert first_k + remaining == full_ids


def test_cutset_state_dict_eager_raises():
    """Eager CutSet raises RuntimeError on state_dict / load_state_dict."""
    from lhotse import CutSet
    from lhotse.testing.dummies import DummyManifest

    cuts = DummyManifest(CutSet, begin_id=0, end_id=5)
    with pytest.raises(RuntimeError, match="lazy"):
        cuts.state_dict()
    with pytest.raises(RuntimeError, match="lazy"):
        cuts.load_state_dict({})
