"""
E2E checkpoint/restore tests for the full dataloading pipeline:

    IterableDatasetWrapper → DynamicBucketingSampler → CutSet

Each test verifies the core property::

    first_k_batches + remaining_batches == all_batches

i.e., checkpointing mid-epoch and restoring on a fresh pipeline produces
the exact same sequence of batches as an uninterrupted run.

Tests cover both the direct IterableDatasetWrapper path and the
torchdata StatefulDataLoader path (when torchdata is installed).
"""

import random

import pytest
import torch.utils.data

from lhotse import CutSet
from lhotse.dataset.cut_transforms import PerturbSpeed, PerturbVolume
from lhotse.dataset.iterable_dataset import IterableDatasetWrapper
from lhotse.dataset.sampling.dynamic_bucketing import DynamicBucketingSampler
from lhotse.testing.dummies import DummyManifest

try:
    from torchdata.stateful_dataloader import StatefulDataLoader

    _HAS_TORCHDATA = True
except ImportError:
    _HAS_TORCHDATA = False

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _IdentityDataset(torch.utils.data.Dataset):
    """Pass-through: returns the sampled CutSet batch as-is."""

    def __getitem__(self, batch):
        return batch


@pytest.fixture()
def cuts_a():
    return DummyManifest(CutSet, begin_id=0, end_id=30)


@pytest.fixture()
def cuts_b():
    return DummyManifest(CutSet, begin_id=100, end_id=130)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_basic_mux_pipeline(cuts_a, cuts_b):
    """filter → repeat(2) → mux — simplest checkpoint/restore."""
    n_consumed = 5

    def make():
        a = cuts_a.filter(lambda c: int(c.id.split("-")[-1]) % 2 == 0).repeat(times=2)
        b = cuts_b.filter(lambda c: int(c.id.split("-")[-1]) % 2 == 1).repeat(times=2)
        pipeline = CutSet.mux(a, b, weights=[0.3, 0.7], seed=42)
        sampler = DynamicBucketingSampler(
            pipeline, max_cuts=5, shuffle=False, seed=0, num_buckets=2
        )
        return IterableDatasetWrapper(_IdentityDataset(), sampler)

    all_batches = [[c.id for c in b] for b in make()]
    assert len(all_batches) > n_consumed

    w1 = make()
    it1 = iter(w1)
    first_k = [[c.id for c in next(it1)] for _ in range(n_consumed)]
    sd = w1.state_dict()

    w2 = make()
    w2.load_state_dict(sd)
    remaining = [[c.id for c in b] for b in w2]

    assert first_k + remaining == all_batches


def test_with_resample(cuts_a, cuts_b):
    """Mux pipeline + resample 16 kHz → 24 kHz."""
    n_consumed = 5

    def make():
        a = cuts_a.filter(lambda c: int(c.id.split("-")[-1]) % 2 == 0).repeat(times=2)
        b = cuts_b.filter(lambda c: int(c.id.split("-")[-1]) % 2 == 1).repeat(times=2)
        pipeline = CutSet.mux(a, b, weights=[0.3, 0.7], seed=42).resample(24000)
        sampler = DynamicBucketingSampler(
            pipeline, max_cuts=5, shuffle=False, seed=0, num_buckets=2
        )
        return IterableDatasetWrapper(_IdentityDataset(), sampler)

    all_batches = [[c.id for c in b] for b in make()]
    assert len(all_batches) > n_consumed

    w1 = make()
    it1 = iter(w1)
    first_k = [[c.id for c in next(it1)] for _ in range(n_consumed)]
    sd = w1.state_dict()

    w2 = make()
    w2.load_state_dict(sd)
    remaining = [[c.id for c in b] for b in w2]

    assert first_k + remaining == all_batches


def test_with_sampler_level_augmentation(cuts_a, cuts_b):
    """Mux pipeline + PerturbSpeed + PerturbVolume applied at the sampler level.

    Verifies that transform RNG states are captured in state_dict() and
    restored in load_state_dict(), so augmentation decisions (which cuts
    get perturbed, which factors are chosen) are identical after restore.
    """
    n_consumed = 5

    def make():
        a = cuts_a.filter(lambda c: int(c.id.split("-")[-1]) % 2 == 0).repeat(times=2)
        b = cuts_b.filter(lambda c: int(c.id.split("-")[-1]) % 2 == 1).repeat(times=2)
        pipeline = CutSet.mux(a, b, weights=[0.3, 0.7], seed=42)
        sampler = DynamicBucketingSampler(
            pipeline, max_cuts=5, shuffle=False, seed=0, num_buckets=2
        )
        sampler.map(PerturbSpeed(factors=[0.9, 1.1], p=0.3, randgen=random.Random(7)))
        sampler.map(PerturbVolume(p=0.2, randgen=random.Random(13)))
        return IterableDatasetWrapper(_IdentityDataset(), sampler)

    all_batches = [[c.id for c in b] for b in make()]
    assert len(all_batches) > n_consumed

    w1 = make()
    it1 = iter(w1)
    first_k = [[c.id for c in next(it1)] for _ in range(n_consumed)]
    sd = w1.state_dict()

    w2 = make()
    w2.load_state_dict(sd)
    remaining = [[c.id for c in b] for b in w2]

    assert first_k + remaining == all_batches


def test_with_mix(cuts_a, cuts_b):
    """Mux pipeline + resample + additive noise mixing (p=0.5).

    LazyCutMixer is not a StatefulIterator so the sampler falls back
    to O(N) fast-forward — this test verifies that path works.
    """
    n_consumed = 5
    noise = DummyManifest(CutSet, begin_id=1000, end_id=1010)

    def make():
        a = cuts_a.filter(lambda c: int(c.id.split("-")[-1]) % 2 == 0).repeat(times=2)
        b = cuts_b.filter(lambda c: int(c.id.split("-")[-1]) % 2 == 1).repeat(times=2)
        pipeline = CutSet.mux(a, b, weights=[0.3, 0.7], seed=42).resample(24000)
        pipeline = pipeline.mix(
            noise.resample(24000), mix_prob=0.5, seed=42, preserve_id="left"
        )
        sampler = DynamicBucketingSampler(
            pipeline, max_cuts=5, shuffle=False, seed=0, num_buckets=2
        )
        return IterableDatasetWrapper(_IdentityDataset(), sampler)

    all_batches = [[c.id for c in b] for b in make()]
    assert len(all_batches) > n_consumed

    w1 = make()
    it1 = iter(w1)
    first_k = [[c.id for c in next(it1)] for _ in range(n_consumed)]
    sd = w1.state_dict()

    w2 = make()
    w2.load_state_dict(sd)
    remaining = [[c.id for c in b] for b in w2]

    assert first_k + remaining == all_batches


def test_full_pipeline(cuts_a, cuts_b):
    """Kitchen-sink: mux + resample + noise mix + sampler-level augmentation."""
    n_consumed = 5
    noise = DummyManifest(CutSet, begin_id=1000, end_id=1010)

    def make():
        a = cuts_a.filter(lambda c: int(c.id.split("-")[-1]) % 2 == 0).repeat(times=2)
        b = cuts_b.filter(lambda c: int(c.id.split("-")[-1]) % 2 == 1).repeat(times=2)
        pipeline = CutSet.mux(a, b, weights=[0.3, 0.7], seed=42).resample(24000)
        pipeline = pipeline.mix(
            noise.resample(24000), mix_prob=0.5, seed=42, preserve_id="left"
        )
        sampler = DynamicBucketingSampler(
            pipeline, max_cuts=5, shuffle=False, seed=0, num_buckets=2
        )
        sampler.map(PerturbSpeed(factors=[0.9, 1.1], p=0.3, randgen=random.Random(7)))
        sampler.map(PerturbVolume(p=0.2, randgen=random.Random(13)))
        return IterableDatasetWrapper(_IdentityDataset(), sampler)

    all_batches = [[c.id for c in b] for b in make()]
    assert len(all_batches) > n_consumed

    w1 = make()
    it1 = iter(w1)
    first_k = [[c.id for c in next(it1)] for _ in range(n_consumed)]
    sd = w1.state_dict()

    w2 = make()
    w2.load_state_dict(sd)
    remaining = [[c.id for c in b] for b in w2]

    assert first_k + remaining == all_batches


@pytest.mark.parametrize("n_consumed", [1, 3, 7])
def test_checkpoint_at_various_positions(cuts_a, cuts_b, n_consumed):
    """Checkpoint at batch 1, 3, and 7 — all produce correct results."""

    def make():
        a = cuts_a.filter(lambda c: int(c.id.split("-")[-1]) % 2 == 0).repeat(times=2)
        b = cuts_b.filter(lambda c: int(c.id.split("-")[-1]) % 2 == 1).repeat(times=2)
        pipeline = CutSet.mux(a, b, weights=[0.3, 0.7], seed=42)
        sampler = DynamicBucketingSampler(
            pipeline, max_cuts=5, shuffle=False, seed=0, num_buckets=2
        )
        return IterableDatasetWrapper(_IdentityDataset(), sampler)

    all_batches = [[c.id for c in b] for b in make()]
    assert len(all_batches) > n_consumed

    w1 = make()
    it1 = iter(w1)
    first_k = [[c.id for c in next(it1)] for _ in range(n_consumed)]
    sd = w1.state_dict()

    w2 = make()
    w2.load_state_dict(sd)
    remaining = [[c.id for c in b] for b in w2]

    assert first_k + remaining == all_batches


@pytest.mark.parametrize("n_consumed", [1, 3, 7])
def test_augmented_checkpoint_at_various_positions(cuts_a, cuts_b, n_consumed):
    """Checkpoint at batch 1, 3, and 7 with sampler-level augmentation."""

    def make():
        a = cuts_a.filter(lambda c: int(c.id.split("-")[-1]) % 2 == 0).repeat(times=2)
        b = cuts_b.filter(lambda c: int(c.id.split("-")[-1]) % 2 == 1).repeat(times=2)
        pipeline = CutSet.mux(a, b, weights=[0.3, 0.7], seed=42)
        sampler = DynamicBucketingSampler(
            pipeline, max_cuts=5, shuffle=False, seed=0, num_buckets=2
        )
        sampler.map(PerturbSpeed(factors=[0.9, 1.1], p=0.3, randgen=random.Random(7)))
        sampler.map(PerturbVolume(p=0.2, randgen=random.Random(13)))
        return IterableDatasetWrapper(_IdentityDataset(), sampler)

    all_batches = [[c.id for c in b] for b in make()]
    assert len(all_batches) > n_consumed

    w1 = make()
    it1 = iter(w1)
    first_k = [[c.id for c in next(it1)] for _ in range(n_consumed)]
    sd = w1.state_dict()

    w2 = make()
    w2.load_state_dict(sd)
    remaining = [[c.id for c in b] for b in w2]

    assert first_k + remaining == all_batches


# ---------------------------------------------------------------------------
# StatefulDataLoader tests (torchdata)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_TORCHDATA, reason="torchdata not installed")
@pytest.mark.parametrize("num_workers", [0, 2])
def test_stateful_dataloader_basic(cuts_a, cuts_b, num_workers):
    """Basic mux pipeline through StatefulDataLoader checkpoint/restore."""
    n_consumed = 3

    def make():
        a = cuts_a.filter(lambda c: int(c.id.split("-")[-1]) % 2 == 0).repeat(times=2)
        b = cuts_b.filter(lambda c: int(c.id.split("-")[-1]) % 2 == 1).repeat(times=2)
        pipeline = CutSet.mux(a, b, weights=[0.3, 0.7], seed=42)
        sampler = DynamicBucketingSampler(
            pipeline, max_cuts=5, shuffle=False, seed=0, num_buckets=2
        )
        return IterableDatasetWrapper(_IdentityDataset(), sampler)

    dl_full = StatefulDataLoader(make(), batch_size=None, num_workers=num_workers)
    all_batches = [[c.id for c in b] for b in dl_full]
    assert len(all_batches) > n_consumed

    dl1 = StatefulDataLoader(make(), batch_size=None, num_workers=num_workers)
    it1 = iter(dl1)
    first_k = [[c.id for c in next(it1)] for _ in range(n_consumed)]
    sd = dl1.state_dict()

    dl2 = StatefulDataLoader(make(), batch_size=None, num_workers=num_workers)
    dl2.load_state_dict(sd)
    remaining = [[c.id for c in b] for b in dl2]

    assert first_k + remaining == all_batches


@pytest.mark.skipif(not _HAS_TORCHDATA, reason="torchdata not installed")
@pytest.mark.parametrize("num_workers", [0, 2])
def test_stateful_dataloader_with_augmentation(cuts_a, cuts_b, num_workers):
    """StatefulDataLoader with sampler-level PerturbSpeed + PerturbVolume."""
    n_consumed = 5

    def make():
        a = cuts_a.filter(lambda c: int(c.id.split("-")[-1]) % 2 == 0).repeat(times=2)
        b = cuts_b.filter(lambda c: int(c.id.split("-")[-1]) % 2 == 1).repeat(times=2)
        pipeline = CutSet.mux(a, b, weights=[0.3, 0.7], seed=42)
        sampler = DynamicBucketingSampler(
            pipeline, max_cuts=5, shuffle=False, seed=0, num_buckets=2
        )
        sampler.map(PerturbSpeed(factors=[0.9, 1.1], p=0.3, randgen=random.Random(7)))
        sampler.map(PerturbVolume(p=0.2, randgen=random.Random(13)))
        return IterableDatasetWrapper(_IdentityDataset(), sampler)

    dl_full = StatefulDataLoader(make(), batch_size=None, num_workers=num_workers)
    all_batches = [[c.id for c in b] for b in dl_full]
    assert len(all_batches) > n_consumed

    dl1 = StatefulDataLoader(make(), batch_size=None, num_workers=num_workers)
    it1 = iter(dl1)
    first_k = [[c.id for c in next(it1)] for _ in range(n_consumed)]
    sd = dl1.state_dict()

    dl2 = StatefulDataLoader(make(), batch_size=None, num_workers=num_workers)
    dl2.load_state_dict(sd)
    remaining = [[c.id for c in b] for b in dl2]

    assert first_k + remaining == all_batches


@pytest.mark.skipif(not _HAS_TORCHDATA, reason="torchdata not installed")
@pytest.mark.parametrize("num_workers", [0, 2])
def test_stateful_dataloader_full_pipeline(cuts_a, cuts_b, num_workers):
    """StatefulDataLoader: mux + resample + noise mix + augmentation."""
    n_consumed = 5
    noise = DummyManifest(CutSet, begin_id=1000, end_id=1010)

    def make():
        a = cuts_a.filter(lambda c: int(c.id.split("-")[-1]) % 2 == 0).repeat(times=2)
        b = cuts_b.filter(lambda c: int(c.id.split("-")[-1]) % 2 == 1).repeat(times=2)
        pipeline = CutSet.mux(a, b, weights=[0.3, 0.7], seed=42).resample(24000)
        pipeline = pipeline.mix(
            noise.resample(24000), mix_prob=0.5, seed=42, preserve_id="left"
        )
        sampler = DynamicBucketingSampler(
            pipeline, max_cuts=5, shuffle=False, seed=0, num_buckets=2
        )
        sampler.map(PerturbSpeed(factors=[0.9, 1.1], p=0.3, randgen=random.Random(7)))
        sampler.map(PerturbVolume(p=0.2, randgen=random.Random(13)))
        return IterableDatasetWrapper(_IdentityDataset(), sampler)

    dl_full = StatefulDataLoader(make(), batch_size=None, num_workers=num_workers)
    all_batches = [[c.id for c in b] for b in dl_full]
    assert len(all_batches) > n_consumed

    dl1 = StatefulDataLoader(make(), batch_size=None, num_workers=num_workers)
    it1 = iter(dl1)
    first_k = [[c.id for c in next(it1)] for _ in range(n_consumed)]
    sd = dl1.state_dict()

    dl2 = StatefulDataLoader(make(), batch_size=None, num_workers=num_workers)
    dl2.load_state_dict(sd)
    remaining = [[c.id for c in b] for b in dl2]

    assert first_k + remaining == all_batches


@pytest.mark.skipif(not _HAS_TORCHDATA, reason="torchdata not installed")
@pytest.mark.parametrize("num_workers", [0, 2])
@pytest.mark.parametrize("n_consumed", [1, 3, 7])
def test_stateful_dataloader_checkpoint_at_various_positions(
    cuts_a, cuts_b, n_consumed, num_workers
):
    """StatefulDataLoader checkpoint at batch 1, 3, and 7."""

    def make():
        a = cuts_a.filter(lambda c: int(c.id.split("-")[-1]) % 2 == 0).repeat(times=2)
        b = cuts_b.filter(lambda c: int(c.id.split("-")[-1]) % 2 == 1).repeat(times=2)
        pipeline = CutSet.mux(a, b, weights=[0.3, 0.7], seed=42)
        sampler = DynamicBucketingSampler(
            pipeline, max_cuts=5, shuffle=False, seed=0, num_buckets=2
        )
        sampler.map(PerturbSpeed(factors=[0.9, 1.1], p=0.3, randgen=random.Random(7)))
        sampler.map(PerturbVolume(p=0.2, randgen=random.Random(13)))
        return IterableDatasetWrapper(_IdentityDataset(), sampler)

    dl_full = StatefulDataLoader(make(), batch_size=None, num_workers=num_workers)
    all_batches = [[c.id for c in b] for b in dl_full]
    assert len(all_batches) > n_consumed

    dl1 = StatefulDataLoader(make(), batch_size=None, num_workers=num_workers)
    it1 = iter(dl1)
    first_k = [[c.id for c in next(it1)] for _ in range(n_consumed)]
    sd = dl1.state_dict()

    dl2 = StatefulDataLoader(make(), batch_size=None, num_workers=num_workers)
    dl2.load_state_dict(sd)
    remaining = [[c.id for c in b] for b in dl2]

    assert first_k + remaining == all_batches
