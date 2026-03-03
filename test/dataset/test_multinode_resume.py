"""
E2E test: resumable multi-node training with StatefulDataLoader.

Simulates the typical Lhotse training setup:

- Two CutSets are made infinite via ``.repeat()`` and blended 70/30 with
  ``CutSet.mux(weights=[0.7, 0.3], seed="randomized")``.  Epochs are
  undefined — the pipeline produces a weighted infinite stream.
- The sampler uses ``rank=0, world_size=1`` (defaults) because Lhotse
  relies on shuffled indexed iterators for high-quality randomization,
  not the sampler's batch-deduplication mechanism.  Every worker iterates
  over the full data independently.
- ``seed="randomized"`` is resolved lazily inside each DataLoader worker
  process via ``make_worker_init_fn(rank=..., world_size=...)``.  This
  gives every ``(dp_rank, worker_id)`` tuple a unique seed, ensuring
  different shuffle orders without explicit coordination.
- We stop after a fixed step budget, checkpoint with
  ``StatefulDataLoader.state_dict()``, and verify that restoration
  produces exactly the same continuation.

Core property::

    first_k_batches + remaining_batches == all_batches

Requires ``torchdata`` (``pip install torchdata``).
"""

import pytest

from lhotse import CutSet
from lhotse.dataset.dataloading import make_worker_init_fn
from lhotse.dataset.iterable_dataset import IdentityDataset, IterableDatasetWrapper
from lhotse.dataset.sampling.dynamic_bucketing import DynamicBucketingSampler
from lhotse.testing.dummies import DummyManifest

try:
    from torchdata.stateful_dataloader import StatefulDataLoader

    _HAS_TORCHDATA = True
except ImportError:
    _HAS_TORCHDATA = False

pytestmark = pytest.mark.skipif(not _HAS_TORCHDATA, reason="torchdata not installed")

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

TOTAL_STEPS = 15
WORLD_SIZE = 2
NUM_WORKERS = 2


def _write_cuts(tmp_path, name, begin, end):
    path = tmp_path / f"{name}.jsonl"
    DummyManifest(CutSet, begin_id=begin, end_id=end).to_jsonl(path)
    return path


def _make_pipeline(cuts_a_path, cuts_b_path):
    """
    Build a fresh infinite mux pipeline.

    ``seed="randomized"`` means each (dp_rank, worker_id) will resolve to
    a different integer seed inside the DataLoader worker subprocess,
    provided ``make_worker_init_fn`` is used.
    """
    a = CutSet.from_file(cuts_a_path, indexed=True).repeat()
    b = CutSet.from_file(cuts_b_path, indexed=True).repeat()
    pipeline = CutSet.mux(a, b, weights=[0.7, 0.3], seed="randomized")
    sampler = DynamicBucketingSampler(
        pipeline,
        max_cuts=4,
        shuffle=True,
        seed="randomized",
        num_buckets=2,
    )
    return IterableDatasetWrapper(IdentityDataset(), sampler)


def _make_dataloader(cuts_a_path, cuts_b_path, dp_rank):
    """Create a StatefulDataLoader for a given data-parallel rank."""
    return StatefulDataLoader(
        _make_pipeline(cuts_a_path, cuts_b_path),
        batch_size=None,
        num_workers=NUM_WORKERS,
        worker_init_fn=make_worker_init_fn(rank=dp_rank, world_size=WORLD_SIZE),
    )


def _take_n(dataloader, n):
    """Consume exactly *n* batches from *dataloader*."""
    it = iter(dataloader)
    return [[c.id for c in next(it)] for _ in range(n)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_all_rank_worker_pairs_see_different_data(tmp_path):
    """
    Every (dp_rank, worker_id) iterates the data in a different order.

    We verify this by collecting cut IDs from each rank's dataloader
    and checking they differ.  Within a rank, the two workers also get
    different seeds via make_worker_init_fn, so batches coming from
    different workers are interleaved with different shuffle orders.
    """
    a_path = _write_cuts(tmp_path, "train_en", begin=0, end=30)
    b_path = _write_cuts(tmp_path, "train_zh", begin=100, end=130)

    per_rank_batches = {}
    for rank in range(WORLD_SIZE):
        dl = _make_dataloader(a_path, b_path, dp_rank=rank)
        per_rank_batches[rank] = _take_n(dl, TOTAL_STEPS)

    # Ranks must see different batch sequences
    assert (
        per_rank_batches[0] != per_rank_batches[1]
    ), "rank 0 and rank 1 should produce different batch sequences"


def test_checkpoint_restore_per_rank(tmp_path):
    """
    Checkpoint mid-training on each rank independently;
    restored run produces the exact same continuation.
    """
    a_path = _write_cuts(tmp_path, "train_en", begin=0, end=30)
    b_path = _write_cuts(tmp_path, "train_zh", begin=100, end=130)
    checkpoint_after = 5

    for rank in range(WORLD_SIZE):
        # Full uninterrupted run
        all_batches = _take_n(
            _make_dataloader(a_path, b_path, dp_rank=rank), TOTAL_STEPS
        )

        # Interrupted: consume checkpoint_after steps, then save
        dl1 = _make_dataloader(a_path, b_path, dp_rank=rank)
        first_k = _take_n(dl1, checkpoint_after)
        sd = dl1.state_dict()

        # Restored: load checkpoint on a fresh pipeline, consume the rest
        dl2 = _make_dataloader(a_path, b_path, dp_rank=rank)
        dl2.load_state_dict(sd)
        remaining = _take_n(dl2, TOTAL_STEPS - checkpoint_after)

        assert (
            first_k + remaining == all_batches
        ), f"rank {rank}: checkpoint/restore mismatch"


def test_both_sources_sampled(tmp_path):
    """Over enough steps, cuts from both CutSets appear in every rank."""
    a_path = _write_cuts(tmp_path, "train_en", begin=0, end=20)
    b_path = _write_cuts(tmp_path, "train_zh", begin=100, end=120)

    for rank in range(WORLD_SIZE):
        dl = _make_dataloader(a_path, b_path, dp_rank=rank)
        all_ids = {cid for batch in _take_n(dl, TOTAL_STEPS) for cid in batch}

        has_a = any("cut-00" in cid for cid in all_ids)
        has_b = any("cut-01" in cid for cid in all_ids)
        assert (
            has_a and has_b
        ), f"rank {rank}: expected cuts from both sources, got {all_ids}"


@pytest.mark.parametrize("checkpoint_after", [1, 7, 12])
def test_checkpoint_at_various_positions(tmp_path, checkpoint_after):
    """Checkpoint at step 1, 7, and 12 — all resume correctly on both ranks."""
    a_path = _write_cuts(tmp_path, "train_en", begin=0, end=30)
    b_path = _write_cuts(tmp_path, "train_zh", begin=100, end=130)

    for rank in range(WORLD_SIZE):
        all_batches = _take_n(
            _make_dataloader(a_path, b_path, dp_rank=rank), TOTAL_STEPS
        )

        dl1 = _make_dataloader(a_path, b_path, dp_rank=rank)
        first_k = _take_n(dl1, checkpoint_after)
        sd = dl1.state_dict()

        dl2 = _make_dataloader(a_path, b_path, dp_rank=rank)
        dl2.load_state_dict(sd)
        remaining = _take_n(dl2, TOTAL_STEPS - checkpoint_after)

        assert (
            first_k + remaining == all_batches
        ), f"rank {rank}, checkpoint_after={checkpoint_after}: mismatch"
