# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Regression tests: ``LazyIndexedSharIterator`` must produce disjoint
slices across DP ranks (and workers) so that an iterable-mode dataloader
doesn't yield the same cut on multiple ranks.

The bug this guards against: prior to integrating
``PartitionedIndexedIterator``, the indexed shar reader skipped the
worker-partition step unless ``split_for_dataloading=True`` was passed
explicitly. NeMo's ``CutSet.from_shar`` call sites never set that flag,
so every DP rank iterated the full cut set. The 0909 validator caught
4072 cross-rank cut duplications attributable solely to AMI lhotse_shar
sources; this file asserts the contract at the iterator level so the
next refactor can't quietly regress it.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import pytest

from lhotse import CutSet
from lhotse.dataset.dataloading import LHOTSE_USE_WORKER_PARTITION
from lhotse.shar.readers.indexed import LazyIndexedSharIterator
from lhotse.shar.writers import SharWriter
from lhotse.testing.dummies import DummyManifest


_PARTITION_ENV_KEYS = ("RANK", "WORLD_SIZE", LHOTSE_USE_WORKER_PARTITION)


@contextmanager
def _env_partition(rank: int, world_size: int):
    """Mimic the worker-subprocess env that ``worker_init_fn`` sets."""
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


_ALL_FIELDS = {
    "recording": "wav",
    "features": "numpy",
    "custom_embedding": "numpy",
    "custom_features": "numpy",
    "custom_indexes": "numpy",
    "custom_recording": "wav",
}


@pytest.fixture
def indexed_shar_dir(tmp_path) -> Path:
    """16 cuts across 4 indexed shards (4 cuts per shard)."""
    cuts = DummyManifest(CutSet, begin_id=0, end_id=16, with_data=True)
    writer = SharWriter(
        tmp_path,
        fields=_ALL_FIELDS,
        shard_size=4,
        compress_jsonl=False,
        create_index=True,
    )
    with writer:
        for c in cuts:
            writer.write(c)
    return tmp_path


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_indexed_shar_partition_disjoint_and_complete(indexed_shar_dir, world_size):
    """Per-rank slices are pairwise disjoint and their union covers everything."""
    expected = {c.id for c in LazyIndexedSharIterator(in_dir=indexed_shar_dir)}
    assert len(expected) == 16

    per_rank: list[set] = []
    for rank in range(world_size):
        with _env_partition(rank=rank, world_size=world_size):
            ids = {c.id for c in LazyIndexedSharIterator(in_dir=indexed_shar_dir)}
        for prev in per_rank:
            assert prev.isdisjoint(ids), (
                f"rank {rank} slice overlaps prior rank: {sorted(prev & ids)}"
            )
        per_rank.append(ids)

    union: set = set()
    for s in per_rank:
        union |= s
    assert union == expected, (
        f"union of {world_size} rank slices misses cuts: "
        f"missing={sorted(expected - union)}, extra={sorted(union - expected)}"
    )


def test_indexed_shar_partition_collapses_without_env_var(indexed_shar_dir):
    """Outside a DataLoader worker (env var unset), partition collapses to (0, 1)
    so map-style mode keeps yielding every cut."""
    expected = {c.id for c in LazyIndexedSharIterator(in_dir=indexed_shar_dir)}
    # Simulate map-style: RANK/WORLD_SIZE may be set by torchrun but the
    # LHOTSE_USE_WORKER_PARTITION gate is OFF.
    saved = {k: os.environ.pop(k, None) for k in _PARTITION_ENV_KEYS}
    os.environ["RANK"] = "3"
    os.environ["WORLD_SIZE"] = "4"
    try:
        ids = {c.id for c in LazyIndexedSharIterator(in_dir=indexed_shar_dir)}
        assert ids == expected
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_indexed_shar_resume_same_topology(indexed_shar_dir):
    """State saved mid-iteration restores correctly under the same (rank, world_size)."""
    with _env_partition(rank=1, world_size=2):
        it1 = LazyIndexedSharIterator(in_dir=indexed_shar_dir)
        all_ids = [c.id for c in LazyIndexedSharIterator(in_dir=indexed_shar_dir)]

        gen = iter(it1)
        consumed = [next(gen).id for _ in range(3)]
        state = it1.state_dict()
        remaining_live = [c.id for c in gen]

        it2 = LazyIndexedSharIterator(in_dir=indexed_shar_dir)
        it2.load_state_dict(state)
        remaining_restored = [c.id for c in it2]

    assert remaining_restored == remaining_live
    assert consumed + remaining_live == all_ids


def test_indexed_shar_resume_topology_mismatch_raises(indexed_shar_dir):
    """Loading a state captured at world_size=4 under world_size=2 must raise.

    Per-shard index sequences would diverge under a different topology;
    silently continuing would corrupt the resumed run's coverage. The
    PartitionedIndexedIterator contract records (shard_id, num_shards)
    at save time and rejects mismatches at load time."""
    with _env_partition(rank=0, world_size=4):
        it1 = LazyIndexedSharIterator(in_dir=indexed_shar_dir)
        gen = iter(it1)
        next(gen)
        state = it1.state_dict()

    with _env_partition(rank=0, world_size=2):
        it2 = LazyIndexedSharIterator(in_dir=indexed_shar_dir)
        it2.load_state_dict(state)
        with pytest.raises(ValueError, match="topology mismatch"):
            list(it2)


@pytest.mark.parametrize("shuffle", [False, True])
def test_indexed_shar_partition_works_with_shuffle(indexed_shar_dir, shuffle):
    """Partition correctness holds whether or not in-iterator shuffle is enabled."""
    expected = {c.id for c in LazyIndexedSharIterator(in_dir=indexed_shar_dir)}
    world_size = 4
    union: set = set()
    for rank in range(world_size):
        with _env_partition(rank=rank, world_size=world_size):
            ids = {c.id for c in LazyIndexedSharIterator(
                in_dir=indexed_shar_dir, shuffle=shuffle, seed=42,
            )}
        assert union.isdisjoint(ids), (
            f"rank {rank} overlaps: {sorted(union & ids)} (shuffle={shuffle})"
        )
        union |= ids
    assert union == expected
