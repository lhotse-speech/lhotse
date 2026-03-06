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

Tests cover:
- Direct IterableDatasetWrapper path (O(N) fast-forward)
- torchdata StatefulDataLoader path (when torchdata is installed)
- O(1) indexed restore via DynamicBucketer state save/restore
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

    With indexed noise input, LazyCutMixer is stateful and supports O(1)
    checkpoint restore.
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


@pytest.mark.skipif(not _HAS_TORCHDATA, reason="torchdata not installed")
def test_stateful_dataloader_worker_prefetch_snapshot_restores_exactly(
    cuts_a_path, cuts_b_path
):
    """Checkpoint after one consumed batch with worker prefetch asymmetry.

    This exercises a snapshot where one worker already yielded a batch and
    carries bucketer state, while another worker is still pre-yield with zero
    diagnostics. Restoring from such a checkpoint must still resume exactly
    from the next global batch.
    """
    n_consumed = 1

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

    dl_full = StatefulDataLoader(make(), batch_size=None, num_workers=2)
    all_batches = [[c.id for c in b] for b in dl_full]
    assert len(all_batches) > n_consumed

    dl1 = StatefulDataLoader(make(), batch_size=None, num_workers=2)
    it1 = iter(dl1)
    first_k = [[c.id for c in next(it1)] for _ in range(n_consumed)]
    sd = dl1.state_dict()

    # Worker snapshots should include a mixed prefetch state:
    # at least one worker with saved bucketer state and at least one worker
    # still at zero yielded batches without bucketer_state.
    worker_snapshots = sd["_snapshot"]["_worker_snapshots"]
    has_post_yield_worker = False
    has_pre_yield_worker = False
    for ws in worker_snapshots.values():
        sampler_state = ws["fetcher_state"]["dataset_iter_state"]["sampler_state"]
        diagnostics = sampler_state["diagnostics"]["stats_per_epoch"][0]
        kept_batches = diagnostics["kept_batches"]
        has_bucketer_state = "bucketer_state" in sampler_state
        if kept_batches > 0 and has_bucketer_state:
            has_post_yield_worker = True
        if kept_batches == 0 and not has_bucketer_state:
            has_pre_yield_worker = True
    assert has_post_yield_worker and has_pre_yield_worker

    dl2 = StatefulDataLoader(make(), batch_size=None, num_workers=2)
    dl2.load_state_dict(sd)
    remaining = [[c.id for c in b] for b in dl2]

    assert first_k + remaining == all_batches
    # Explicitly guard against replaying the already-consumed first batch.
    assert remaining[0] == all_batches[1]


# ---------------------------------------------------------------------------
# O(1) indexed restore tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_consumed", [1, 3, 5])
def test_indexed_o1_restore(cuts_a_path, cuts_b_path, n_consumed):
    """O(1) indexed restore produces same results as uninterrupted run."""

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

    # Verify that bucketer_state is captured in the state_dict
    sampler_sd = sd.get("sampler_state", sd)
    assert (
        "bucketer_state" in sampler_sd and "rng_state" in sampler_sd
    ), "O(1) indexed restore keys should be in state_dict for indexed datasets"

    w2 = make()
    w2.load_state_dict(sd)
    remaining = [[c.id for c in b] for b in w2]

    assert first_k + remaining == all_batches


def test_indexed_o1_restore_with_augmentation(cuts_a_path, cuts_b_path):
    """O(1) indexed restore with sampler-level augmentation."""
    n_consumed = 3

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
# Shar fields-based E2E tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_consumed", [1, 3])
def test_shar_fields_o1_restore(tmp_path, n_consumed):
    """O(1) indexed restore works with fields-based LazySharIterator."""
    from lhotse.shar.readers.lazy import LazySharIterator

    path_a = tmp_path / "cuts_a.000000.jsonl"
    path_b = tmp_path / "cuts_b.000000.jsonl"
    DummyManifest(CutSet, begin_id=0, end_id=30).to_jsonl(path_a)
    DummyManifest(CutSet, begin_id=100, end_id=130).to_jsonl(path_b)

    def make():
        a = CutSet(LazySharIterator(fields={"cuts": [str(path_a)]}))
        b = CutSet(LazySharIterator(fields={"cuts": [str(path_b)]}))
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
