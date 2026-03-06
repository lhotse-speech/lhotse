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
- Direct IterableDatasetWrapper path
- torchdata StatefulDataLoader path (when torchdata is installed)
- O(1) indexed restore via DynamicBucketer state save/restore
"""

import random
from copy import deepcopy

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


def _make_mux_pipeline(cuts_a_path, cuts_b_path, *, resample_to=None, noise_path=None):
    a = _load(cuts_a_path).filter(_even_filter).repeat(times=2)
    b = _load(cuts_b_path).filter(_odd_filter).repeat(times=2)
    pipeline = CutSet.mux(a, b, weights=[0.3, 0.7], seed=42)
    if resample_to is not None:
        pipeline = pipeline.resample(resample_to)
    if noise_path is not None:
        noise = _load(noise_path)
        if resample_to is not None:
            noise = noise.resample(resample_to)
        pipeline = pipeline.mix(
            noise,
            mix_prob=0.5,
            seed=42,
            preserve_id="left",
        )
    return pipeline


def _apply_sampler_augmentation(sampler):
    sampler.map(PerturbSpeed(factors=[0.9, 1.1], p=0.3, randgen=random.Random(7)))
    sampler.map(PerturbVolume(p=0.2, randgen=random.Random(13)))


def _make_wrapper(pipeline, *, augment=False):
    sampler = DynamicBucketingSampler(
        pipeline, max_cuts=5, shuffle=False, seed=0, num_buckets=2
    )
    if augment:
        _apply_sampler_augmentation(sampler)
    return IterableDatasetWrapper(_IdentityDataset(), sampler)


def _batch_ids(batch):
    return [cut.id for cut in batch]


def _consume_batches(iterator, n_consumed):
    return [_batch_ids(next(iterator)) for _ in range(n_consumed)]


def _collect_batches(iterable):
    return [_batch_ids(batch) for batch in iterable]


def _run_wrapper_checkpoint(make_wrapper, n_consumed):
    all_batches = _collect_batches(make_wrapper())
    assert len(all_batches) > n_consumed

    wrapper = make_wrapper()
    first_k = _consume_batches(iter(wrapper), n_consumed)
    state = wrapper.state_dict()
    checkpoint = deepcopy(state)

    restored = make_wrapper()
    restored.load_state_dict(state)
    remaining = _collect_batches(restored)
    return all_batches, first_k, remaining, checkpoint


def _assert_wrapper_restore(make_wrapper, n_consumed, *, assert_state=None):
    all_batches, first_k, remaining, state = _run_wrapper_checkpoint(
        make_wrapper, n_consumed
    )
    if assert_state is not None:
        assert_state(state)
    assert first_k + remaining == all_batches


def _run_stateful_dataloader_checkpoint(make_wrapper, n_consumed, num_workers):
    full = StatefulDataLoader(make_wrapper(), batch_size=None, num_workers=num_workers)
    all_batches = _collect_batches(full)
    assert len(all_batches) > n_consumed

    dloader = StatefulDataLoader(
        make_wrapper(), batch_size=None, num_workers=num_workers
    )
    first_k = _consume_batches(iter(dloader), n_consumed)
    state = dloader.state_dict()
    checkpoint = deepcopy(state)

    restored = StatefulDataLoader(
        make_wrapper(), batch_size=None, num_workers=num_workers
    )
    restored.load_state_dict(state)
    remaining = _collect_batches(restored)
    return all_batches, first_k, remaining, checkpoint


def _assert_stateful_dataloader_restore(make_wrapper, n_consumed, num_workers):
    all_batches, first_k, remaining, _ = _run_stateful_dataloader_checkpoint(
        make_wrapper, n_consumed, num_workers
    )
    assert first_k + remaining == all_batches


# ---------------------------------------------------------------------------
# IterableDatasetWrapper tests
# ---------------------------------------------------------------------------


def test_basic_mux_pipeline(cuts_a_path, cuts_b_path):
    """filter → repeat(2) → mux — simplest checkpoint/restore."""

    def make():
        return _make_wrapper(_make_mux_pipeline(cuts_a_path, cuts_b_path))

    _assert_wrapper_restore(make, n_consumed=5)


def test_with_resample(cuts_a_path, cuts_b_path):
    """Mux pipeline + resample 16 kHz → 24 kHz."""

    def make():
        return _make_wrapper(
            _make_mux_pipeline(cuts_a_path, cuts_b_path, resample_to=24000)
        )

    _assert_wrapper_restore(make, n_consumed=5)


def test_with_sampler_level_augmentation(cuts_a_path, cuts_b_path):
    """Mux pipeline + PerturbSpeed + PerturbVolume applied at the sampler level.

    Verifies that transform RNG states are captured in state_dict() and
    restored in load_state_dict(), so augmentation decisions (which cuts
    get perturbed, which factors are chosen) are identical after restore.
    """

    def make():
        return _make_wrapper(_make_mux_pipeline(cuts_a_path, cuts_b_path), augment=True)

    _assert_wrapper_restore(make, n_consumed=5)


def test_with_mix(cuts_a_path, cuts_b_path, tmp_path):
    """Mux pipeline + resample + additive noise mixing (p=0.5).

    With indexed noise input, LazyCutMixer is stateful and supports O(1)
    checkpoint restore.
    """
    noise_path = tmp_path / "noise.jsonl"
    DummyManifest(CutSet, begin_id=1000, end_id=1010).to_jsonl(noise_path)

    def make():
        return _make_wrapper(
            _make_mux_pipeline(
                cuts_a_path,
                cuts_b_path,
                resample_to=24000,
                noise_path=noise_path,
            )
        )

    _assert_wrapper_restore(make, n_consumed=5)


def test_full_pipeline(cuts_a_path, cuts_b_path, tmp_path):
    """Kitchen-sink: mux + resample + noise mix + sampler-level augmentation."""
    noise_path = tmp_path / "noise.jsonl"
    DummyManifest(CutSet, begin_id=1000, end_id=1010).to_jsonl(noise_path)

    def make():
        return _make_wrapper(
            _make_mux_pipeline(
                cuts_a_path,
                cuts_b_path,
                resample_to=24000,
                noise_path=noise_path,
            ),
            augment=True,
        )

    _assert_wrapper_restore(make, n_consumed=5)


@pytest.mark.parametrize("n_consumed", [1, 3, 7])
def test_checkpoint_at_various_positions(cuts_a_path, cuts_b_path, n_consumed):
    """Checkpoint at batch 1, 3, and 7 — all produce correct results."""

    def make():
        return _make_wrapper(_make_mux_pipeline(cuts_a_path, cuts_b_path))

    _assert_wrapper_restore(make, n_consumed=n_consumed)


@pytest.mark.parametrize("n_consumed", [1, 3, 7])
def test_augmented_checkpoint_at_various_positions(
    cuts_a_path, cuts_b_path, n_consumed
):
    """Checkpoint at batch 1, 3, and 7 with sampler-level augmentation."""

    def make():
        return _make_wrapper(_make_mux_pipeline(cuts_a_path, cuts_b_path), augment=True)

    _assert_wrapper_restore(make, n_consumed=n_consumed)


# ---------------------------------------------------------------------------
# StatefulDataLoader tests (torchdata)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_TORCHDATA, reason="torchdata not installed")
@pytest.mark.parametrize("num_workers", [0, 2])
def test_stateful_dataloader_basic(cuts_a_path, cuts_b_path, num_workers):
    """Basic mux pipeline through StatefulDataLoader checkpoint/restore."""

    def make():
        return _make_wrapper(_make_mux_pipeline(cuts_a_path, cuts_b_path))

    _assert_stateful_dataloader_restore(make, n_consumed=3, num_workers=num_workers)


@pytest.mark.skipif(not _HAS_TORCHDATA, reason="torchdata not installed")
@pytest.mark.parametrize("num_workers", [0, 2])
def test_stateful_dataloader_with_augmentation(cuts_a_path, cuts_b_path, num_workers):
    """StatefulDataLoader with sampler-level PerturbSpeed + PerturbVolume."""

    def make():
        return _make_wrapper(_make_mux_pipeline(cuts_a_path, cuts_b_path), augment=True)

    _assert_stateful_dataloader_restore(make, n_consumed=5, num_workers=num_workers)


@pytest.mark.skipif(not _HAS_TORCHDATA, reason="torchdata not installed")
@pytest.mark.parametrize("num_workers", [0, 2])
def test_stateful_dataloader_full_pipeline(
    cuts_a_path, cuts_b_path, tmp_path, num_workers
):
    """StatefulDataLoader: mux + resample + noise mix + augmentation."""
    noise_path = tmp_path / "noise.jsonl"
    DummyManifest(CutSet, begin_id=1000, end_id=1010).to_jsonl(noise_path)

    def make():
        return _make_wrapper(
            _make_mux_pipeline(
                cuts_a_path,
                cuts_b_path,
                resample_to=24000,
                noise_path=noise_path,
            ),
            augment=True,
        )

    _assert_stateful_dataloader_restore(make, n_consumed=5, num_workers=num_workers)


@pytest.mark.skipif(not _HAS_TORCHDATA, reason="torchdata not installed")
@pytest.mark.parametrize("num_workers", [0, 2])
@pytest.mark.parametrize("n_consumed", [1, 3, 7])
def test_stateful_dataloader_checkpoint_at_various_positions(
    cuts_a_path, cuts_b_path, n_consumed, num_workers
):
    """StatefulDataLoader checkpoint at batch 1, 3, and 7."""

    def make():
        return _make_wrapper(_make_mux_pipeline(cuts_a_path, cuts_b_path), augment=True)

    _assert_stateful_dataloader_restore(
        make, n_consumed=n_consumed, num_workers=num_workers
    )


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

    def make():
        return _make_wrapper(_make_mux_pipeline(cuts_a_path, cuts_b_path), augment=True)

    all_batches, first_k, remaining, sd = _run_stateful_dataloader_checkpoint(
        make, n_consumed=1, num_workers=2
    )

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
        return _make_wrapper(_make_mux_pipeline(cuts_a_path, cuts_b_path))

    def assert_state(state):
        sampler_sd = state.get("sampler_state", state)
        assert "bucketer_state" in sampler_sd and "rng_state" in sampler_sd

    _assert_wrapper_restore(make, n_consumed=n_consumed, assert_state=assert_state)


def test_indexed_o1_restore_with_augmentation(cuts_a_path, cuts_b_path):
    """O(1) indexed restore with sampler-level augmentation."""

    def make():
        return _make_wrapper(_make_mux_pipeline(cuts_a_path, cuts_b_path), augment=True)

    _assert_wrapper_restore(make, n_consumed=3)


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
        return _make_wrapper(CutSet.mux(a, b, weights=[0.3, 0.7], seed=42))

    _assert_wrapper_restore(make, n_consumed=n_consumed)
