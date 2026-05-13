"""
Tests for (DP rank x DataLoader worker) sample-level partitioning
in the indexed-iterator path.

The partition is realized by :class:`lhotse.indexing.LazyShuffledRange`,
plumbed through :class:`lhotse.lazy.LazyIndexedManifestIterator` and
:class:`lhotse.lazy.LazyIteratorChain` (globally-shuffled path), and
resolved at iter time inside DataLoader worker subprocesses via
:func:`lhotse.dataset.dataloading.get_worker_partition`.

These tests simulate the (rank, world_size) env-var setup that
``worker_init_fn`` provides at runtime and verify that the union of
items yielded by all shards equals the full manifest, with no duplicates.
"""

import os
from contextlib import contextmanager

import pytest

from lhotse import CutSet
from lhotse.dataset.dataloading import (
    LHOTSE_USE_WORKER_PARTITION,
    PartitionedIndexedIterator,
    get_worker_partition,
    worker_init_fn,
)
from lhotse.indexing import LazyShuffledRange, create_jsonl_index
from lhotse.lazy import (
    LazyFilter,
    LazyIndexedManifestIterator,
    LazyIteratorChain,
    LazyIteratorMultiplexer,
    LazyMapper,
    LazyRepeater,
    LazyShuffler,
)
from lhotse.testing.dummies import DummyManifest


_PARTITION_ENV_KEYS = ("RANK", "WORLD_SIZE", LHOTSE_USE_WORKER_PARTITION)


@contextmanager
def _env_partition(rank: int, world_size: int):
    """Simulate the worker-process env that ``worker_init_fn`` sets: RANK,
    WORLD_SIZE, plus the ``LHOTSE_USE_WORKER_PARTITION`` signal."""
    saved = {k: os.environ.get(k) for k in _PARTITION_ENV_KEYS}
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ[LHOTSE_USE_WORKER_PARTITION] = "1"
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextmanager
def _env_map_style(rank: int, world_size: int):
    """Simulate map-style mode: torchrun sets RANK/WORLD_SIZE in the main
    process, but ``worker_init_fn`` is never called so
    ``LHOTSE_USE_WORKER_PARTITION`` stays unset. Partition must collapse to
    ``(0, 1)`` so the sampler's over-sample-and-discard works correctly."""
    saved = {k: os.environ.get(k) for k in _PARTITION_ENV_KEYS}
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ.pop(LHOTSE_USE_WORKER_PARTITION, None)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.fixture
def indexed_jsonl(tmp_path):
    p = tmp_path / "data.jsonl"
    DummyManifest(CutSet, begin_id=0, end_id=50).to_jsonl(p)
    create_jsonl_index(p)
    return p


@pytest.fixture
def two_indexed_jsonls(tmp_path):
    p1 = tmp_path / "a.jsonl"
    p2 = tmp_path / "b.jsonl"
    DummyManifest(CutSet, begin_id=0, end_id=20).to_jsonl(p1)
    DummyManifest(CutSet, begin_id=100, end_id=130).to_jsonl(p2)
    create_jsonl_index(p1)
    create_jsonl_index(p2)
    return p1, p2


# ---------------------------------------------------------------------------
# get_worker_partition
# ---------------------------------------------------------------------------


def test_get_worker_partition_main_process_no_env():
    """Without env vars and outside a worker, partition is the trivial (0, 1)."""
    saved = {k: os.environ.pop(k, None) for k in _PARTITION_ENV_KEYS}
    try:
        assert get_worker_partition() == (0, 1)
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def test_get_worker_partition_uses_env_rank():
    with _env_partition(rank=3, world_size=8):
        # num_workers=0 outside a DataLoader → worker_id=0, num_workers=1.
        assert get_worker_partition() == (3, 8)


def test_get_worker_partition_ignores_rank_without_signal():
    """RANK/WORLD_SIZE set without LHOTSE_USE_WORKER_PARTITION (e.g. torchrun
    + map-style mode where worker_init_fn never fires): partition is (0, 1)."""
    with _env_map_style(rank=2, world_size=4):
        assert get_worker_partition() == (0, 1)


def test_worker_init_fn_eager_call_sets_env_vars():
    """Eager main-process call (num_workers=0 path) wires env vars correctly."""
    saved = {k: os.environ.pop(k, None) for k in _PARTITION_ENV_KEYS}
    try:
        worker_init_fn(0, rank=2, world_size=5, seed=42)
        assert os.environ["RANK"] == "2"
        assert os.environ["WORLD_SIZE"] == "5"
        assert os.environ[LHOTSE_USE_WORKER_PARTITION] == "1"
        assert get_worker_partition() == (2, 5)
    finally:
        for k in _PARTITION_ENV_KEYS:
            os.environ.pop(k, None)
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


# ---------------------------------------------------------------------------
# LazyIndexedManifestIterator partition
# ---------------------------------------------------------------------------


def _all_expected_ids(begin: int, end: int) -> list:
    return [f"dummy-mono-cut-{i:04d}" for i in range(begin, end)]


@pytest.mark.parametrize("world_size", [2, 3, 4, 7])
def test_indexed_manifest_iterator_partition_shuffled(indexed_jsonl, world_size):
    """All shards together yield every item exactly once (shuffle=True)."""
    seen_ids = []
    for rank in range(world_size):
        with _env_partition(rank=rank, world_size=world_size):
            it = LazyIndexedManifestIterator(indexed_jsonl, shuffle=True, seed=7)
            for item in it:
                seen_ids.append(item.id)
    assert sorted(seen_ids) == sorted(_all_expected_ids(0, 50))


@pytest.mark.parametrize("world_size", [2, 3, 4, 7])
def test_indexed_manifest_iterator_partition_non_shuffled(indexed_jsonl, world_size):
    """Non-shuffled iteration also partitions by sample index (stride)."""
    seen_ids = []
    for rank in range(world_size):
        with _env_partition(rank=rank, world_size=world_size):
            it = LazyIndexedManifestIterator(indexed_jsonl, shuffle=False)
            for item in it:
                seen_ids.append(item.id)
    assert sorted(seen_ids) == sorted(_all_expected_ids(0, 50))


def test_indexed_manifest_iterator_partition_default_matches_unpartitioned(indexed_jsonl):
    """Without env vars (single-shard default), behaviour matches the un-sharded path."""
    saved = {k: os.environ.pop(k, None) for k in _PARTITION_ENV_KEYS}
    try:
        it_a = LazyIndexedManifestIterator(indexed_jsonl, shuffle=True, seed=11)
        a = [item.id for item in it_a]
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v

    with _env_partition(rank=0, world_size=1):
        it_b = LazyIndexedManifestIterator(indexed_jsonl, shuffle=True, seed=11)
        b = [item.id for item in it_b]

    assert a == b
    assert sorted(a) == sorted(_all_expected_ids(0, 50))


def test_indexed_manifest_iterator_partition_state_dict_roundtrip(indexed_jsonl):
    """Resume mid-shard yields the same remaining items as an uninterrupted run."""
    with _env_partition(rank=1, world_size=4):
        it_full = LazyIndexedManifestIterator(indexed_jsonl, shuffle=True, seed=99)
        all_items = [item.id for item in it_full]

        it_interrupted = LazyIndexedManifestIterator(indexed_jsonl, shuffle=True, seed=99)
        gen = iter(it_interrupted)
        first_part = [next(gen).id for _ in range(min(4, len(all_items) // 2))]
        sd = it_interrupted.state_dict()

        it_restored = LazyIndexedManifestIterator(indexed_jsonl, shuffle=True, seed=99)
        it_restored.load_state_dict(sd)
        remaining = [item.id for item in it_restored]

        assert first_part + remaining == all_items


def test_indexed_manifest_iterator_partition_resume_topology_mismatch_raises(indexed_jsonl):
    """Saving at one shard topology and restoring at another raises ValueError."""
    with _env_partition(rank=1, world_size=4):
        it = LazyIndexedManifestIterator(indexed_jsonl, shuffle=True, seed=99)
        gen = iter(it)
        _ = next(gen)
        sd = it.state_dict()

    with _env_partition(rank=2, world_size=8):
        it2 = LazyIndexedManifestIterator(indexed_jsonl, shuffle=True, seed=99)
        it2.load_state_dict(sd)
        with pytest.raises(ValueError, match="topology mismatch"):
            list(it2)


# ---------------------------------------------------------------------------
# LazyIteratorChain (globally-shuffled) partition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("world_size", [2, 3, 5])
def test_chain_globally_shuffled_partition(two_indexed_jsonls, world_size):
    """Chain globally-shuffled iteration partitions the combined index range."""
    p1, p2 = two_indexed_jsonls
    expected = set(_all_expected_ids(0, 20)) | set(_all_expected_ids(100, 130))
    seen = []
    for rank in range(world_size):
        with _env_partition(rank=rank, world_size=world_size):
            chain = LazyIteratorChain(
                LazyIndexedManifestIterator(p1),
                LazyIndexedManifestIterator(p2),
                shuffle_iters=True,
                seed=13,
            )
            for item in chain:
                seen.append(item.id)
    assert sorted(seen) == sorted(expected)


def test_chain_globally_shuffled_no_double_partition_small(tmp_path):
    """Regression test for the specific concern: 2 ranks x 2 chained manifests of 4 items each.

    Chain's _iter_globally_shuffled builds a partitioned LazyShuffledRange over the
    *combined* index range and routes items via __getitem__, which bypasses the source's
    own partitioning. So the chain partition is applied exactly once. With 2 ranks each
    rank must see 4 items (total/world_size), with the union across ranks covering all 8."""
    p1 = tmp_path / "a.jsonl"
    p2 = tmp_path / "b.jsonl"
    DummyManifest(CutSet, begin_id=0, end_id=4).to_jsonl(p1)
    DummyManifest(CutSet, begin_id=100, end_id=104).to_jsonl(p2)
    create_jsonl_index(p1)
    create_jsonl_index(p2)
    expected = set(_all_expected_ids(0, 4)) | set(_all_expected_ids(100, 104))
    assert len(expected) == 8

    seen = []
    per_rank_counts = []
    for rank in range(2):
        with _env_partition(rank=rank, world_size=2):
            chain = LazyIteratorChain(
                LazyIndexedManifestIterator(p1),
                LazyIndexedManifestIterator(p2),
                shuffle_iters=True,
                seed=13,
            )
            rank_items = [item.id for item in chain]
            per_rank_counts.append(len(rank_items))
            seen.extend(rank_items)
    assert per_rank_counts == [4, 4], f"Expected each rank to see 4 items, got {per_rank_counts}"
    assert sorted(seen) == sorted(expected)
    assert len(seen) == len(set(seen)) == 8


@pytest.mark.parametrize("world_size", [2, 3, 5])
def test_chain_sequential_partition_indexed_sources(two_indexed_jsonls, world_size):
    """Chain in sequential mode (shuffle_iters=False) lets each indexed source partition
    itself. No chain-level partition; single source-level partition applied per source."""
    p1, p2 = two_indexed_jsonls
    expected = set(_all_expected_ids(0, 20)) | set(_all_expected_ids(100, 130))
    seen = []
    for rank in range(world_size):
        with _env_partition(rank=rank, world_size=world_size):
            chain = LazyIteratorChain(
                LazyIndexedManifestIterator(p1),
                LazyIndexedManifestIterator(p2),
                shuffle_iters=False,
            )
            for item in chain:
                seen.append(item.id)
    assert sorted(seen) == sorted(expected)
    assert len(seen) == len(set(seen)) == 50


def test_chain_globally_shuffled_no_double_partition_uneven_sources(tmp_path):
    """Chain partition still covers every item when sources have uneven sizes
    (3 + 5 = 8 items, 2 ranks) — Feistel permutation is bijective over the combined range."""
    p1 = tmp_path / "a.jsonl"
    p2 = tmp_path / "b.jsonl"
    DummyManifest(CutSet, begin_id=0, end_id=3).to_jsonl(p1)
    DummyManifest(CutSet, begin_id=100, end_id=105).to_jsonl(p2)
    create_jsonl_index(p1)
    create_jsonl_index(p2)
    expected = set(_all_expected_ids(0, 3)) | set(_all_expected_ids(100, 105))

    seen = []
    for rank in range(2):
        with _env_partition(rank=rank, world_size=2):
            chain = LazyIteratorChain(
                LazyIndexedManifestIterator(p1),
                LazyIndexedManifestIterator(p2),
                shuffle_iters=True,
                seed=21,
            )
            for item in chain:
                seen.append(item.id)
    assert sorted(seen) == sorted(expected)
    assert len(seen) == len(set(seen)) == 8


# ---------------------------------------------------------------------------
# LazyIteratorMultiplexer seed assertion
# ---------------------------------------------------------------------------


def test_multiplexer_rejects_randomized_seed_under_multishard(two_indexed_jsonls):
    """LazyIteratorMultiplexer with seed='randomized' raises under multi-shard."""
    p1, p2 = two_indexed_jsonls
    with _env_partition(rank=0, world_size=4):
        mux = LazyIteratorMultiplexer(
            LazyIndexedManifestIterator(p1),
            LazyIndexedManifestIterator(p2),
            seed="randomized",
        )
        with pytest.raises(ValueError, match="seed='randomized'"):
            next(iter(mux))


def test_multiplexer_allows_randomized_seed_single_shard(two_indexed_jsonls):
    """Single-shard (no DDP, num_workers=0) still accepts seed='randomized'."""
    p1, p2 = two_indexed_jsonls
    saved = {k: os.environ.pop(k, None) for k in _PARTITION_ENV_KEYS}
    try:
        mux = LazyIteratorMultiplexer(
            LazyIndexedManifestIterator(p1),
            LazyIndexedManifestIterator(p2),
            seed="randomized",
        )
        items = list(mux)
        assert len(items) == 50
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def test_multiplexer_partition_two_sources_equal_size_small(tmp_path):
    """User's exact scenario for the multiplexer: 2 manifests of 4 items each, 2 ranks.

    Each source's __iter__ applies its own partition; the multiplexer picks
    among sources with a fixed-seed RNG so all ranks agree on the source-pick
    sequence. Each rank sees 4 items; union across ranks covers all 8."""
    p1 = tmp_path / "a.jsonl"
    p2 = tmp_path / "b.jsonl"
    DummyManifest(CutSet, begin_id=0, end_id=4).to_jsonl(p1)
    DummyManifest(CutSet, begin_id=100, end_id=104).to_jsonl(p2)
    create_jsonl_index(p1)
    create_jsonl_index(p2)
    expected = set(_all_expected_ids(0, 4)) | set(_all_expected_ids(100, 104))
    assert len(expected) == 8

    seen = []
    per_rank_counts = []
    for rank in range(2):
        with _env_partition(rank=rank, world_size=2):
            mux = LazyIteratorMultiplexer(
                LazyIndexedManifestIterator(p1),
                LazyIndexedManifestIterator(p2),
                seed=42,
            )
            rank_items = [item.id for item in mux]
            per_rank_counts.append(len(rank_items))
            seen.extend(rank_items)
    assert per_rank_counts == [4, 4], f"Expected each rank to see 4 items, got {per_rank_counts}"
    assert sorted(seen) == sorted(expected)
    assert len(seen) == len(set(seen)) == 8


@pytest.mark.parametrize("world_size", [2, 3, 5])
def test_multiplexer_partition_full_coverage(two_indexed_jsonls, world_size):
    """Multiplexer over partitioned sources: union of items across all ranks
    equals the full manifest, no duplicates. Tests both equal- and uneven-shard cases."""
    p1, p2 = two_indexed_jsonls
    expected = set(_all_expected_ids(0, 20)) | set(_all_expected_ids(100, 130))
    seen = []
    for rank in range(world_size):
        with _env_partition(rank=rank, world_size=world_size):
            mux = LazyIteratorMultiplexer(
                LazyIndexedManifestIterator(p1),
                LazyIndexedManifestIterator(p2),
                seed=7,
            )
            for item in mux:
                seen.append(item.id)
    assert sorted(seen) == sorted(expected)
    assert len(seen) == len(set(seen)) == 50


def test_multiplexer_partition_uneven_shard_sizes(tmp_path):
    """Uneven shard sizes (one rank exhausts before another) still cover everything.

    M0=5 items: rank 0 shard has 3, rank 1 shard has 2.
    M1=7 items: rank 0 shard has 4, rank 1 shard has 3.
    Rank 1 exhausts its M0 shard sooner than rank 0; from that step on the ranks'
    RNG states for source-selection diverge, but each rank fully consumes its own
    shards before marking sources exhausted. Union must still cover everything."""
    p1 = tmp_path / "a.jsonl"
    p2 = tmp_path / "b.jsonl"
    DummyManifest(CutSet, begin_id=0, end_id=5).to_jsonl(p1)
    DummyManifest(CutSet, begin_id=100, end_id=107).to_jsonl(p2)
    create_jsonl_index(p1)
    create_jsonl_index(p2)
    expected = set(_all_expected_ids(0, 5)) | set(_all_expected_ids(100, 107))
    assert len(expected) == 12

    seen = []
    per_rank_counts = []
    for rank in range(2):
        with _env_partition(rank=rank, world_size=2):
            mux = LazyIteratorMultiplexer(
                LazyIndexedManifestIterator(p1),
                LazyIndexedManifestIterator(p2),
                seed=123,
            )
            rank_items = [item.id for item in mux]
            per_rank_counts.append(len(rank_items))
            seen.extend(rank_items)
    # Each rank yields its full share of items from each source.
    assert per_rank_counts[0] == 3 + 4, f"rank 0 expected 7 items, got {per_rank_counts[0]}"
    assert per_rank_counts[1] == 2 + 3, f"rank 1 expected 5 items, got {per_rank_counts[1]}"
    assert sorted(seen) == sorted(expected)
    assert len(seen) == len(set(seen)) == 12


def test_multiplexer_partition_with_weights(two_indexed_jsonls):
    """Weighted multiplexer with partition: each item still appears exactly once
    across ranks; weights only bias the global mux pick sequence, not coverage."""
    p1, p2 = two_indexed_jsonls
    expected = set(_all_expected_ids(0, 20)) | set(_all_expected_ids(100, 130))

    seen = []
    for rank in range(4):
        with _env_partition(rank=rank, world_size=4):
            mux = LazyIteratorMultiplexer(
                LazyIndexedManifestIterator(p1),
                LazyIndexedManifestIterator(p2),
                weights=[0.7, 0.3],
                seed=99,
            )
            for item in mux:
                seen.append(item.id)
    assert sorted(seen) == sorted(expected)
    assert len(seen) == len(set(seen)) == 50


def test_multiplexer_state_dict_roundtrip_under_partition(tmp_path):
    """Multiplexer state captures per-rank RNG and per-rank exhausted-source mask;
    after restore, the rank's remaining items match an uninterrupted run."""
    p1 = tmp_path / "a.jsonl"
    p2 = tmp_path / "b.jsonl"
    DummyManifest(CutSet, begin_id=0, end_id=10).to_jsonl(p1)
    DummyManifest(CutSet, begin_id=100, end_id=110).to_jsonl(p2)
    create_jsonl_index(p1)
    create_jsonl_index(p2)

    def make():
        return LazyIteratorMultiplexer(
            LazyIndexedManifestIterator(p1),
            LazyIndexedManifestIterator(p2),
            seed=42,
        )

    with _env_partition(rank=0, world_size=2):
        mux_full = make()
        all_items = [item.id for item in mux_full]

        mux1 = make()
        gen1 = iter(mux1)
        first_part = [next(gen1).id for _ in range(3)]
        sd = mux1.state_dict()

        mux2 = make()
        mux2.load_state_dict(sd)
        remaining = [item.id for item in mux2]

        assert first_part + remaining == all_items


# ---------------------------------------------------------------------------
# Map-style regression — partition must NOT apply
# ---------------------------------------------------------------------------


def test_map_style_path_yields_all_items_under_torchrun(indexed_jsonl):
    """Regression: in map-style mode (torchrun sets RANK/WORLD_SIZE in the main
    process, no worker_init_fn ever fires), the source iterator must yield ALL
    items — DP dedup is the sampler's job via over-sample-and-discard.

    Before this fix the iterator would have collapsed to 1/world_size of the data."""
    with _env_map_style(rank=2, world_size=4):
        it = LazyIndexedManifestIterator(indexed_jsonl, shuffle=True, seed=42)
        items = [item.id for item in it]
        assert len(items) == 50
        assert sorted(items) == sorted(_all_expected_ids(0, 50))


def test_map_style_path_non_shuffled_yields_all_items(indexed_jsonl):
    """Same regression check for the non-shuffled stride path."""
    with _env_map_style(rank=1, world_size=8):
        it = LazyIndexedManifestIterator(indexed_jsonl, shuffle=False)
        items = [item.id for item in it]
        assert len(items) == 50
        assert sorted(items) == sorted(_all_expected_ids(0, 50))


def test_map_style_chain_globally_shuffled_yields_all_items(two_indexed_jsonls):
    """Chain global-shuffle path also collapses to no-partition in map-style mode."""
    p1, p2 = two_indexed_jsonls
    with _env_map_style(rank=3, world_size=4):
        chain = LazyIteratorChain(
            LazyIndexedManifestIterator(p1),
            LazyIndexedManifestIterator(p2),
            shuffle_iters=True,
            seed=7,
        )
        items = [item.id for item in chain]
        assert len(items) == 50


def test_map_style_multiplexer_yields_all_items_no_seed_check(two_indexed_jsonls):
    """LazyIteratorMultiplexer with seed='randomized' is rejected only under
    multi-shard partition. In map-style mode, partition is (0, 1) so randomized
    seeds work as before."""
    p1, p2 = two_indexed_jsonls
    with _env_map_style(rank=0, world_size=2):
        mux = LazyIteratorMultiplexer(
            LazyIndexedManifestIterator(p1),
            LazyIndexedManifestIterator(p2),
            seed="randomized",
        )
        items = list(mux)
        assert len(items) == 50


# ---------------------------------------------------------------------------
# Empty / undersized manifests
# ---------------------------------------------------------------------------


def test_partition_empty_manifest(tmp_path):
    """Empty manifest: every shard yields zero items, no errors."""
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    create_jsonl_index(p)
    for rank in range(3):
        with _env_partition(rank=rank, world_size=3):
            it = LazyIndexedManifestIterator(p, shuffle=True, seed=0)
            assert list(it) == []
            it2 = LazyIndexedManifestIterator(p, shuffle=False)
            assert list(it2) == []


def test_partition_n_less_than_num_shards_some_shards_empty(tmp_path):
    """When n < num_shards, some shards must be empty but iteration completes."""
    p = tmp_path / "tiny.jsonl"
    DummyManifest(CutSet, begin_id=0, end_id=3).to_jsonl(p)
    create_jsonl_index(p)

    seen = []
    per_rank_counts = []
    for rank in range(5):  # 5 shards, only 3 items
        with _env_partition(rank=rank, world_size=5):
            it = LazyIndexedManifestIterator(p, shuffle=True, seed=0)
            rank_items = [item.id for item in it]
            per_rank_counts.append(len(rank_items))
            seen.extend(rank_items)
    # Three shards get 1 item each; two shards get 0.
    assert sum(per_rank_counts) == 3
    assert per_rank_counts.count(0) == 2
    assert per_rank_counts.count(1) == 3
    assert sorted(seen) == sorted(_all_expected_ids(0, 3))


def test_partition_n_equals_num_shards(tmp_path):
    """When n == num_shards, each shard gets exactly 1 item."""
    p = tmp_path / "equal.jsonl"
    DummyManifest(CutSet, begin_id=0, end_id=4).to_jsonl(p)
    create_jsonl_index(p)
    seen = []
    for rank in range(4):
        with _env_partition(rank=rank, world_size=4):
            it = LazyIndexedManifestIterator(p, shuffle=True, seed=99)
            rank_items = [item.id for item in it]
            assert len(rank_items) == 1
            seen.extend(rank_items)
    assert sorted(seen) == sorted(_all_expected_ids(0, 4))


# ---------------------------------------------------------------------------
# Composition with other lazy iterator nodes
# ---------------------------------------------------------------------------


def test_lazy_shuffler_over_partitioned_indexed(indexed_jsonl):
    """LazyShuffler buffering of partitioned items: each rank's shuffler sees only
    its shard; reshuffling within the shard preserves coverage across ranks."""
    seen = []
    for rank in range(4):
        with _env_partition(rank=rank, world_size=4):
            shuffler = LazyShuffler(
                LazyIndexedManifestIterator(indexed_jsonl, shuffle=True, seed=7),
                buffer_size=5,
            )
            seen.extend(item.id for item in shuffler)
    assert sorted(seen) == sorted(_all_expected_ids(0, 50))
    assert len(seen) == len(set(seen)) == 50


def test_lazy_mapper_over_partitioned_indexed(indexed_jsonl):
    """LazyMapper applies a transform to each partitioned item; coverage preserved."""

    def add_tag(cut):
        cut.tagged = True
        return cut

    seen = []
    for rank in range(3):
        with _env_partition(rank=rank, world_size=3):
            mapper = LazyMapper(
                LazyIndexedManifestIterator(indexed_jsonl, shuffle=True, seed=11),
                fn=add_tag,
            )
            for item in mapper:
                assert getattr(item, "tagged", False) is True
                seen.append(item.id)
    assert sorted(seen) == sorted(_all_expected_ids(0, 50))
    assert len(seen) == len(set(seen)) == 50


def test_lazy_filter_over_partitioned_indexed(indexed_jsonl):
    """LazyFilter drops items by predicate; partition + filter still composes
    so that union of accepted items across ranks == full filter applied globally."""

    def even_index(cut):
        return int(cut.id.split("-")[-1]) % 2 == 0

    expected = sorted(c for c in _all_expected_ids(0, 50) if int(c.split("-")[-1]) % 2 == 0)
    seen = []
    for rank in range(4):
        with _env_partition(rank=rank, world_size=4):
            filt = LazyFilter(
                LazyIndexedManifestIterator(indexed_jsonl, shuffle=True, seed=11),
                predicate=even_index,
            )
            seen.extend(item.id for item in filt)
    assert sorted(seen) == expected
    assert len(seen) == len(set(seen))


def test_lazy_repeater_over_partitioned_indexed(indexed_jsonl):
    """LazyRepeater re-iterates a partitioned source N times.

    Since LazyIndexedManifestIterator uses a fixed seed for its LazyShuffledRange,
    each epoch yields the *same* partitioned items in the *same* order. The repeater
    therefore yields shard_len items per epoch, totalling times * shard_len.
    Repeater appends a ``_repeat{N}`` suffix to cut IDs each pass; strip it to
    compare item identity rather than the augmented ID."""
    times = 3

    def strip_repeat(cut_id: str) -> str:
        return cut_id.rsplit("_repeat", 1)[0]

    for rank in range(2):
        with _env_partition(rank=rank, world_size=2):
            source = LazyIndexedManifestIterator(indexed_jsonl, shuffle=True, seed=5)
            shard_len_expected = (50 - rank + 1) // 2
            single_pass = [item.id for item in source]
            assert len(single_pass) == shard_len_expected

            rep = LazyRepeater(
                LazyIndexedManifestIterator(indexed_jsonl, shuffle=True, seed=5),
                times=times,
            )
            repeated = [strip_repeat(item.id) for item in rep]
            assert len(repeated) == times * shard_len_expected
            for k in range(times):
                assert repeated[k * shard_len_expected : (k + 1) * shard_len_expected] == single_pass


# ---------------------------------------------------------------------------
# Determinism and resume
# ---------------------------------------------------------------------------


def test_partition_determinism_across_runs(indexed_jsonl):
    """Same (rank, world_size, seed) yields the same partition + order across calls."""
    for rank in range(3):
        with _env_partition(rank=rank, world_size=3):
            a = [item.id for item in LazyIndexedManifestIterator(indexed_jsonl, shuffle=True, seed=33)]
            b = [item.id for item in LazyIndexedManifestIterator(indexed_jsonl, shuffle=True, seed=33)]
            assert a == b


def test_partition_different_seeds_different_orders(indexed_jsonl):
    """Same partition with different seeds yields different orderings. Note: the
    *set* of items each rank receives also differs across seeds, because the
    Feistel permutation is a bijection of [0, n) and a different seed maps the
    same stride of logical offsets onto a different subset of physical indices.
    What's invariant is that the union across all ranks always covers [0, n)."""
    a_per_rank = []
    b_per_rank = []
    for rank in range(2):
        with _env_partition(rank=rank, world_size=2):
            a_per_rank.append([item.id for item in LazyIndexedManifestIterator(indexed_jsonl, shuffle=True, seed=1)])
            b_per_rank.append([item.id for item in LazyIndexedManifestIterator(indexed_jsonl, shuffle=True, seed=2)])
    # Rank 0 with seed 1 differs from rank 0 with seed 2.
    assert a_per_rank[0] != b_per_rank[0]
    # But each seed individually still produces a full partition of the manifest.
    assert sorted(a_per_rank[0] + a_per_rank[1]) == sorted(_all_expected_ids(0, 50))
    assert sorted(b_per_rank[0] + b_per_rank[1]) == sorted(_all_expected_ids(0, 50))


def test_chain_globally_shuffled_topology_mismatch_on_resume(two_indexed_jsonls):
    """Saving chain state at one (rank, world_size) and trying to resume at another
    raises with a clear partition-mismatch error."""
    p1, p2 = two_indexed_jsonls

    with _env_partition(rank=0, world_size=2):
        chain = LazyIteratorChain(
            LazyIndexedManifestIterator(p1),
            LazyIndexedManifestIterator(p2),
            shuffle_iters=True,
            seed=13,
        )
        gen = iter(chain)
        _ = next(gen)
        sd = chain.state_dict()

    with _env_partition(rank=1, world_size=4):
        chain2 = LazyIteratorChain(
            LazyIndexedManifestIterator(p1),
            LazyIndexedManifestIterator(p2),
            shuffle_iters=True,
            seed=13,
        )
        chain2.load_state_dict(sd)
        with pytest.raises(ValueError, match="partition mismatch"):
            list(chain2)


# ---------------------------------------------------------------------------
# Worker-id partition (DP rank × DataLoader worker) — simulated
# ---------------------------------------------------------------------------


class _FakeWorkerInfo:
    def __init__(self, id_: int, num_workers: int):
        self.id = id_
        self.num_workers = num_workers


def test_get_worker_partition_combines_rank_and_worker_id(monkeypatch):
    """When called inside a simulated DataLoader worker subprocess, the partition
    combines DP rank with worker_id: shard_id = rank * num_workers + worker_id."""
    import torch.utils.data as tud

    with _env_partition(rank=2, world_size=4):
        # Two workers per rank → 8 total shards.
        monkeypatch.setattr(tud, "get_worker_info", lambda: _FakeWorkerInfo(1, 2))
        assert get_worker_partition() == (2 * 2 + 1, 4 * 2)


def test_get_worker_partition_full_coverage_rank_x_worker(tmp_path, monkeypatch):
    """End-to-end: simulate 2 ranks × 3 workers each = 6 shards over a 30-item
    manifest. Union across all (rank, worker) tuples is the full manifest."""
    import torch.utils.data as tud

    p = tmp_path / "data.jsonl"
    DummyManifest(CutSet, begin_id=0, end_id=30).to_jsonl(p)
    create_jsonl_index(p)

    seen = []
    for rank in range(2):
        for worker_id in range(3):
            with _env_partition(rank=rank, world_size=2):
                monkeypatch.setattr(tud, "get_worker_info", lambda wid=worker_id: _FakeWorkerInfo(wid, 3))
                it = LazyIndexedManifestIterator(p, shuffle=True, seed=21)
                seen.extend(item.id for item in it)
    assert sorted(seen) == sorted(_all_expected_ids(0, 30))
    assert len(seen) == len(set(seen)) == 30


# ---------------------------------------------------------------------------
# Mixed indexed / non-indexed sources — documented asymmetry
# ---------------------------------------------------------------------------


def test_chain_mixed_indexed_non_indexed_only_indexed_partitions(tmp_path):
    """Documented behaviour: in a chain mixing indexed + non-indexed sources, only
    indexed sources receive partition. Non-indexed (LazyManifestIterator backed by
    LazyJsonlIterator) keeps its pre-partition behaviour, so every rank reads its
    items in full — they are *not* deduplicated across ranks. Users with mixed
    setups must rely on other dedup mechanisms (e.g. shard pre-splitting) for the
    non-indexed parts.

    The chain falls back to ``_iter_sequential`` because ``is_indexed`` is False
    when any source is non-indexed."""
    from lhotse.lazy import LazyManifestIterator

    p1 = tmp_path / "indexed.jsonl"
    p2 = tmp_path / "plain.jsonl"
    DummyManifest(CutSet, begin_id=0, end_id=8).to_jsonl(p1)
    DummyManifest(CutSet, begin_id=100, end_id=108).to_jsonl(p2)
    create_jsonl_index(p1)
    # p2 has no index — use LazyManifestIterator (non-indexed, deserializes to Cut).

    per_rank_indexed = []
    per_rank_plain = []
    for rank in range(2):
        with _env_partition(rank=rank, world_size=2):
            chain = LazyIteratorChain(
                LazyIndexedManifestIterator(p1),
                LazyManifestIterator(p2),
                shuffle_iters=False,
            )
            ids = [item.id for item in chain]
            per_rank_indexed.append([i for i in ids if i.startswith("dummy-mono-cut-0000")
                                                       or i.startswith("dummy-mono-cut-0001")
                                                       or i.startswith("dummy-mono-cut-0002")
                                                       or i.startswith("dummy-mono-cut-0003")
                                                       or i.startswith("dummy-mono-cut-0004")
                                                       or i.startswith("dummy-mono-cut-0005")
                                                       or i.startswith("dummy-mono-cut-0006")
                                                       or i.startswith("dummy-mono-cut-0007")])
            per_rank_plain.append([i for i in ids if int(i.split("-")[-1]) >= 100])
    # Indexed source: each rank gets 4 of 8. Non-indexed source: each rank gets all 8.
    assert len(per_rank_indexed[0]) == 4
    assert len(per_rank_indexed[1]) == 4
    indexed_first_eight = [f"dummy-mono-cut-{i:04d}" for i in range(8)]
    assert sorted(per_rank_indexed[0] + per_rank_indexed[1]) == sorted(indexed_first_eight)
    # Non-indexed half: each rank sees all 8 items from p2 (no partition applied).
    assert sorted(per_rank_plain[0]) == sorted(per_rank_plain[1])
    assert len(per_rank_plain[0]) == 8


# ---------------------------------------------------------------------------
# PartitionedIndexedIterator helper.
#
# These tests pin the contract: full coverage at world_size=1, disjoint slices
# under multi-rank, position-tracked resume, and a strict topology-mismatch
# error on resume. Iterators that delegate to this helper inherit the contract.
# ---------------------------------------------------------------------------


def test_partitioned_iterator_single_rank_full_coverage():
    """world_size=1: yields every global index 0..N-1 in order."""
    it = PartitionedIndexedIterator()
    assert list(it.iterate(10)) == list(range(10))
    sd = it.state_dict()
    assert sd["position"] == 10
    assert sd["shard_id"] == 0
    assert sd["num_shards"] == 1


@pytest.mark.parametrize("world_size", [2, 3, 4, 7])
def test_partitioned_iterator_multi_rank_disjoint_full_coverage(world_size):
    """Every rank gets a disjoint slice; the union equals the full range."""
    n = 50
    per_rank = []
    for rank in range(world_size):
        with _env_partition(rank=rank, world_size=world_size):
            it = PartitionedIndexedIterator()
            per_rank.append(list(it.iterate(n)))
    union = [idx for slc in per_rank for idx in slc]
    assert sorted(union) == list(range(n))
    for r in range(world_size):
        for s in range(r + 1, world_size):
            assert set(per_rank[r]).isdisjoint(per_rank[s])


def test_partitioned_iterator_resume_from_middle():
    """Save position mid-iteration, restore, finish — gets the tail intact."""
    with _env_partition(rank=1, world_size=4):
        it = PartitionedIndexedIterator()
        gen = it.iterate(100)
        first_half = [next(gen) for _ in range(6)]
        sd = it.state_dict()

        # Reload into a fresh iterator (simulating a process restart) and
        # finish the iteration; the two halves should reconstruct the
        # original rank-1 slice exactly.
        it2 = PartitionedIndexedIterator()
        it2.load_state_dict(sd)
        second_half = list(it2.iterate(100))

        ref = PartitionedIndexedIterator()
        full = list(ref.iterate(100))
        assert first_half + second_half == full


def test_partitioned_iterator_resume_topology_mismatch_raises():
    """Saving under (rank=0, world=4) then resuming under (rank=0, world=8)
    must raise — the per-shard sequence would diverge silently otherwise."""
    with _env_partition(rank=0, world_size=4):
        it = PartitionedIndexedIterator()
        gen = it.iterate(64)
        for _ in range(3):
            next(gen)
        sd = it.state_dict()

    with _env_partition(rank=0, world_size=8):
        it2 = PartitionedIndexedIterator()
        it2.load_state_dict(sd)
        with pytest.raises(ValueError, match="topology mismatch"):
            next(it2.iterate(64))


def test_partitioned_iterator_map_style_path_yields_all():
    """Map-style mode: RANK/WORLD_SIZE set but LHOTSE_USE_WORKER_PARTITION
    unset. Partition collapses to (0, 1), helper yields the full range so
    the sampler's own over-sample-and-discard handles DP dedup."""
    with _env_map_style(rank=2, world_size=4):
        it = PartitionedIndexedIterator()
        assert list(it.iterate(25)) == list(range(25))


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_partitioned_iterator_empty_total_len_no_yields(world_size):
    """total_len=0 yields nothing on every rank."""
    for rank in range(world_size):
        with _env_partition(rank=rank, world_size=world_size):
            it = PartitionedIndexedIterator()
            assert list(it.iterate(0)) == []


def test_partitioned_iterator_n_smaller_than_world_size():
    """If total_len < world_size, low ranks get one item, high ranks get none."""
    n, world_size = 3, 8
    per_rank = []
    for rank in range(world_size):
        with _env_partition(rank=rank, world_size=world_size):
            it = PartitionedIndexedIterator()
            per_rank.append(list(it.iterate(n)))
    # First n ranks each get a single index = their rank; the rest are empty.
    assert per_rank[:n] == [[0], [1], [2]]
    assert all(slc == [] for slc in per_rank[n:])


def test_partitioned_iterator_state_dict_before_iter_is_neutral():
    """Initial state_dict carries None topology — a load+iterate cycle must
    not raise on the first run because saved topology is None (= unknown)."""
    src = PartitionedIndexedIterator()
    sd = src.state_dict()
    assert sd == {"position": 0, "shard_id": None, "num_shards": None}

    dst = PartitionedIndexedIterator()
    dst.load_state_dict(sd)
    # No topology recorded → no mismatch error possible; iterates from 0.
    with _env_partition(rank=0, world_size=2):
        assert list(dst.iterate(8)) == [0, 2, 4, 6]
