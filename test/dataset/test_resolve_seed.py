import random

import pytest

from lhotse.dataset.dataloading import LHOTSE_PROCESS_SEED, resolve_seed


@pytest.fixture
def no_rank_env(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv(LHOTSE_PROCESS_SEED, raising=False)


def _resolve_randomized_on_rank(monkeypatch, rank: int, global_seed: int) -> int:
    monkeypatch.setenv("RANK", str(rank))
    random.seed(global_seed)
    return resolve_seed("randomized")


def test_resolve_seed_int_passthrough():
    assert resolve_seed(1234) == 1234


def test_resolve_seed_trng_is_in_range():
    seed = resolve_seed("trng")
    assert isinstance(seed, int)
    assert 0 <= seed < 2**31


@pytest.mark.parametrize("seed", [None, "randomized"])
def test_resolve_seed_without_workers_depends_on_global_seed(no_rank_env, seed):
    # Regression: the first word of the Mersenne Twister state is always 0x80000000
    # after random.seed(...), so this used to return the same value for every global seed.
    random.seed(0)
    seed_0 = resolve_seed(seed)
    random.seed(1)
    seed_1 = resolve_seed(seed)
    assert seed_0 != seed_1
    assert seed_0 != 0x80000000
    assert 0 <= seed_0 < 2**31
    assert 0 <= seed_1 < 2**31


@pytest.mark.parametrize("seed", [None, "randomized"])
def test_resolve_seed_without_workers_is_reproducible_and_side_effect_free(
    no_rank_env, seed
):
    random.seed(42)
    first = resolve_seed(seed)
    second = resolve_seed(seed)
    next_draw = random.random()

    random.seed(42)
    assert resolve_seed(seed) == first
    assert random.random() == next_draw
    assert second == first


def test_resolve_seed_randomized_without_workers_differs_across_ranks(monkeypatch):
    # Regression: with num_workers=0, every DDP rank resolved "randomized"
    # to the same seed, so every rank read the same data.
    monkeypatch.delenv(LHOTSE_PROCESS_SEED, raising=False)
    # Same global seed on every rank, as with seed_everything(42) in DDP.
    seeds = [
        _resolve_randomized_on_rank(monkeypatch, rank=rank, global_seed=42)
        for rank in range(4)
    ]
    assert len(set(seeds)) == 4
    assert all(0 <= s < 2**31 for s in seeds)
    # Still reproducible for a given (global seed, rank).
    assert _resolve_randomized_on_rank(monkeypatch, rank=1, global_seed=42) == seeds[1]


def test_resolve_seed_randomized_in_worker_reads_process_seed(monkeypatch):
    class FakeWorkerInfo:
        id = 0
        num_workers = 2

    monkeypatch.setattr("torch.utils.data.get_worker_info", lambda: FakeWorkerInfo())
    monkeypatch.setenv(LHOTSE_PROCESS_SEED, "100042")
    assert resolve_seed("randomized") == 100042
