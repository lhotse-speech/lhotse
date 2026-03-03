"""
E2E checkpoint/restore tests for the full dataloading pipeline:

    IterableDatasetWrapper → DynamicBucketingSampler → CutSet

Each test verifies the core property::

    first_k_batches + remaining_batches == all_batches

i.e., checkpointing mid-epoch and restoring on a fresh pipeline produces
the exact same sequence of batches as an uninterrupted run.

All data sources are **file-backed indexed lazy JSONL** manifests loaded
via ``CutSet.from_file(path, indexed=True)``, exercising the
``LazyIndexedManifestIterator`` path end-to-end.

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


def _even_filter(c):
    """Keep cuts whose numeric suffix is even."""
    return int(c.id.split("-")[-1]) % 2 == 0


def _odd_filter(c):
    """Keep cuts whose numeric suffix is odd."""
    return int(c.id.split("-")[-1]) % 2 == 1


@pytest.fixture()
def cuts_a_path(tmp_path):
    path = tmp_path / "cuts_a.jsonl"
    DummyManifest(CutSet, begin_id=0, end_id=30).to_jsonl(path)
    return path


@pytest.fixture()
def cuts_b_path(tmp_path):
    path = tmp_path / "cuts_b.jsonl"
    DummyManifest(CutSet, begin_id=100, end_id=130).to_jsonl(path)
    return path


def _load(path):
    """Load an indexed lazy CutSet from a JSONL file."""
    return CutSet.from_file(path, indexed=True)


# ---------------------------------------------------------------------------
# IterableDatasetWrapper tests
# ---------------------------------------------------------------------------


def test_basic_mux_pipeline(cuts_a_path, cuts_b_path):
    """filter → repeat(2) → mux — simplest checkpoint/restore."""
    n_consumed = 5

    def make():
        a = _load(cuts_a_path).filter(_even_filter).repeat(times=2)
        b = _load(cuts_b_path).filter(_odd_filter).repeat(times=2)
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


def test_with_resample(cuts_a_path, cuts_b_path):
    """Mux pipeline + resample 16 kHz → 24 kHz."""
    n_consumed = 5

    def make():
        a = _load(cuts_a_path).filter(_even_filter).repeat(times=2)
        b = _load(cuts_b_path).filter(_odd_filter).repeat(times=2)
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


def test_with_sampler_level_augmentation(cuts_a_path, cuts_b_path):
    """Mux pipeline + PerturbSpeed + PerturbVolume applied at the sampler level.

    Verifies that transform RNG states are captured in state_dict() and
    restored in load_state_dict(), so augmentation decisions (which cuts
    get perturbed, which factors are chosen) are identical after restore.
    """
    n_consumed = 5

    def make():
        a = _load(cuts_a_path).filter(_even_filter).repeat(times=2)
        b = _load(cuts_b_path).filter(_odd_filter).repeat(times=2)
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


def test_with_mix(cuts_a_path, cuts_b_path, tmp_path):
    """Mux pipeline + resample + additive noise mixing (p=0.5).

    LazyCutMixer is not a StatefulIterator so the sampler falls back
    to O(N) fast-forward — this test verifies that path works.
    """
    n_consumed = 5
    noise_path = tmp_path / "noise.jsonl"
    DummyManifest(CutSet, begin_id=1000, end_id=1010).to_jsonl(noise_path)

    def make():
        a = _load(cuts_a_path).filter(_even_filter).repeat(times=2)
        b = _load(cuts_b_path).filter(_odd_filter).repeat(times=2)
        noise = _load(noise_path)
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


def test_full_pipeline(cuts_a_path, cuts_b_path, tmp_path):
    """Kitchen-sink: mux + resample + noise mix + sampler-level augmentation."""
    n_consumed = 5
    noise_path = tmp_path / "noise.jsonl"
    DummyManifest(CutSet, begin_id=1000, end_id=1010).to_jsonl(noise_path)

    def make():
        a = _load(cuts_a_path).filter(_even_filter).repeat(times=2)
        b = _load(cuts_b_path).filter(_odd_filter).repeat(times=2)
        noise = _load(noise_path)
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
def test_checkpoint_at_various_positions(cuts_a_path, cuts_b_path, n_consumed):
    """Checkpoint at batch 1, 3, and 7 — all produce correct results."""

    def make():
        a = _load(cuts_a_path).filter(_even_filter).repeat(times=2)
        b = _load(cuts_b_path).filter(_odd_filter).repeat(times=2)
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
def test_augmented_checkpoint_at_various_positions(
    cuts_a_path, cuts_b_path, n_consumed
):
    """Checkpoint at batch 1, 3, and 7 with sampler-level augmentation."""

    def make():
        a = _load(cuts_a_path).filter(_even_filter).repeat(times=2)
        b = _load(cuts_b_path).filter(_odd_filter).repeat(times=2)
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
def test_stateful_dataloader_basic(cuts_a_path, cuts_b_path, num_workers):
    """Basic mux pipeline through StatefulDataLoader checkpoint/restore."""
    n_consumed = 3

    def make():
        a = _load(cuts_a_path).filter(_even_filter).repeat(times=2)
        b = _load(cuts_b_path).filter(_odd_filter).repeat(times=2)
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
def test_stateful_dataloader_with_augmentation(cuts_a_path, cuts_b_path, num_workers):
    """StatefulDataLoader with sampler-level PerturbSpeed + PerturbVolume."""
    n_consumed = 5

    def make():
        a = _load(cuts_a_path).filter(_even_filter).repeat(times=2)
        b = _load(cuts_b_path).filter(_odd_filter).repeat(times=2)
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
def test_stateful_dataloader_full_pipeline(
    cuts_a_path, cuts_b_path, tmp_path, num_workers
):
    """StatefulDataLoader: mux + resample + noise mix + augmentation."""
    n_consumed = 5
    noise_path = tmp_path / "noise.jsonl"
    DummyManifest(CutSet, begin_id=1000, end_id=1010).to_jsonl(noise_path)

    def make():
        a = _load(cuts_a_path).filter(_even_filter).repeat(times=2)
        b = _load(cuts_b_path).filter(_odd_filter).repeat(times=2)
        noise = _load(noise_path)
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
    cuts_a_path, cuts_b_path, n_consumed, num_workers
):
    """StatefulDataLoader checkpoint at batch 1, 3, and 7."""

    def make():
        a = _load(cuts_a_path).filter(_even_filter).repeat(times=2)
        b = _load(cuts_b_path).filter(_odd_filter).repeat(times=2)
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


# ---------------------------------------------------------------------------
# CutSet state_dict / load_state_dict
# ---------------------------------------------------------------------------


def test_cutset_state_dict_basic(tmp_path):
    """Lazy CutSet round-trip: first_k + remaining == full."""
    from lhotse.lazy import LazyIndexedManifestIterator

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
    path = tmp_path / "cuts.jsonl"
    DummyManifest(CutSet, begin_id=0, end_id=20).to_jsonl(path)

    def make():
        return _load(path).resample(24000)

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
    cuts = DummyManifest(CutSet, begin_id=0, end_id=5)
    with pytest.raises(RuntimeError, match="lazy"):
        cuts.state_dict()
    with pytest.raises(RuntimeError, match="lazy"):
        cuts.load_state_dict({})


def test_cutset_from_file_indexed(tmp_path):
    """from_file(indexed=True) uses indexed iterator, has_constant_time_access == True."""
    path = tmp_path / "cuts.jsonl"
    DummyManifest(CutSet, begin_id=0, end_id=10).to_jsonl(path)

    cs = _load(path)
    assert cs.is_lazy
    assert cs.has_constant_time_access is True
    assert len(list(cs)) == 10


def test_cutset_getitem_indexed(tmp_path):
    """O(1) random access through transform chain on indexed CutSet."""
    from lhotse.lazy import LazyIndexedManifestIterator, LazyMapper

    path = tmp_path / "cuts.jsonl"
    DummyManifest(CutSet, begin_id=0, end_id=10).to_jsonl(path)

    indexed = LazyIndexedManifestIterator(path)
    mapped = LazyMapper(indexed, fn=lambda c: c)
    cs = CutSet(mapped)

    assert cs.has_constant_time_access is True

    # Access via CutSet.__getitem__ (int index)
    c0 = cs[0]
    c5 = cs[5]
    assert c0.id == indexed[0].id
    assert c5.id == indexed[5].id
