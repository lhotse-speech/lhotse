from copy import deepcopy
from types import SimpleNamespace

import pytest

from lhotse import CutSet, load_manifest_lazy
from lhotse.dataset import (
    PackedSequenceDynamicBucketingSampler,
    PackedSequenceDynamicCutSampler,
    PackedTokenConstraint,
)
from lhotse.dataset.iterable_dataset import IdentityDataset, IterableDatasetWrapper
from lhotse.dataset.sampling.packed import _fit_output_cap, _select_best_fit_indices
from lhotse.testing.dummies import DummyManifest, dummy_cut

both_samplers = pytest.mark.parametrize("kind", ["cut", "bucketing"])


@pytest.fixture
def cuts_path(tmp_path):
    def write(num_tokens=(7, 4, 6, 3), num_output_tokens=None, name="cuts.jsonl"):
        cuts = []
        for idx, length in enumerate(num_tokens):
            cut = dummy_cut(idx, duration=float(length))
            cut.custom = {"num_tokens": length}
            if num_output_tokens is not None:
                cut.custom["num_output_tokens"] = num_output_tokens[idx]
            cuts.append(cut)
        path = tmp_path / name
        CutSet.from_cuts(cuts).to_file(path)
        return path

    return write


def make_sampler(
    kind,
    *paths,
    indexed=False,
    packing_buffer_size=4,
    drop_last=False,
    shuffle=False,
    duration_bins=(100,),
    buffer_size=32,
    batch_tokens=10,
    **constraint_kwargs,
):
    cuts = [load_manifest_lazy(path, indexed=indexed) for path in paths]
    kwargs = dict(
        constraint=PackedTokenConstraint(
            batch_tokens=batch_tokens, **constraint_kwargs
        ),
        packing_buffer_size=packing_buffer_size,
        drop_last=drop_last,
        shuffle=shuffle,
        seed=0,
    )
    if kind == "cut":
        return PackedSequenceDynamicCutSampler(*cuts, **kwargs)
    return PackedSequenceDynamicBucketingSampler(
        *cuts, duration_bins=list(duration_bins), buffer_size=buffer_size, **kwargs
    )


def batch_ids(batches):
    return [[int(cut.id[-4:]) for cut in batch] for batch in batches]


def tuple_batch_ids(batches):
    return [tuple(batch_ids(batch)) for batch in batches]


def remaining_batches(sampler):
    """Drain the current epoch; ``iter(sampler)`` would start a new one."""
    batches = []
    while True:
        try:
            batches.append(next(sampler))
        except StopIteration:
            return batches


def test_select_best_fit_indices_is_exact_and_prefers_earlier_items():
    assert _select_best_fit_indices([4, 6, 3, 3, 1], capacity=7) == [0, 2]
    assert _select_best_fit_indices([4, 6, 3, 3, 1], capacity=7, max_items=1) == [1]
    assert _select_best_fit_indices([1, 1, 2], capacity=2) == [0, 1]
    assert _select_best_fit_indices([1, 1, 2], capacity=2, max_items=2) == [2]
    assert _select_best_fit_indices([5, 5], capacity=4) == []


def test_fit_output_cap_drops_largest_then_backfills():
    # The anchor at index 0 leaves 10 budget and 8 output tokens for the tail.
    budgets, outputs = [4, 5, 5, 3, 2], [2, 6, 6, 1, 1]
    caps = dict(batch_tokens=14, output_cap=10)
    # The later of two equally large outputs is dropped, then the small candidates fill both budgets.
    assert _fit_output_cap([0, 1], budgets, outputs, max_items=None, **caps) == [
        0,
        2,
        3,
    ]
    assert _fit_output_cap([0, 1], budgets, outputs, max_items=2, **caps) == [0, 2]
    with pytest.raises(ValueError, match="exceed batch_output_tokens=5"):
        _fit_output_cap([], [4], [6], batch_tokens=14, output_cap=5, max_items=None)


def test_packed_token_constraint():
    constraint = PackedTokenConstraint(
        batch_tokens=100, batch_output_tokens=10, quadratic_length=10
    )
    example = SimpleNamespace(num_tokens=5, num_output_tokens=6)
    assert constraint.measure_packing_length(example) == 8
    constraint.add(example)
    assert (constraint.current, constraint.current_output) == (8, 6)
    assert constraint.close_to_exceeding()
    assert not constraint.exceeded()
    constraint.add(example)
    assert constraint.exceeded()
    constraint.reset()
    assert (constraint.current, constraint.current_output, constraint.num_examples) == (
        0,
        0,
        0,
    )


def test_packed_token_constraint_max_examples_per_bucket():
    constraint = PackedTokenConstraint(
        batch_tokens=100,
        max_examples=3,
        bucket_bins=[5, 10],
        max_examples_per_bucket=[4, 1],
    )
    assert constraint.max_examples_for_length(5) == 3
    assert constraint.max_examples_for_length(6) == 1
    with pytest.raises(ValueError, match="exceeds the highest bucket boundary"):
        constraint.max_examples_for_length(11)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(batch_tokens=0),
        dict(batch_tokens=None),
        dict(batch_tokens=10, max_examples=0),
        dict(batch_tokens=10, bucket_bins=[5]),
        dict(batch_tokens=10, bucket_bins=[5, 3], max_examples_per_bucket=[1, 1]),
    ],
)
def test_packed_token_constraint_rejects_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        PackedTokenConstraint(**kwargs)


@both_samplers
@pytest.mark.parametrize(
    ("num_tokens", "kwargs", "expected_ids"),
    [
        # A greedy batcher stops before the 4/6-token examples; best-fit packing
        # reaches past them for the 3-token example.
        ((7, 4, 6, 3), {}, [[0, 3], [1, 2]]),
        ((9, 12, 4, 3), dict(batch_tokens=16), [[0, 2, 3], [1]]),
        # Example 1 doesn't fit beside example 0, but it anchors the very next batch.
        ((6, 6, 4, 4, 4, 4), {}, [[0, 2], [1, 3], [4, 5]]),
        ((2, 2, 2, 2), dict(max_examples=2), [[0, 1], [2, 3]]),
        # Each 5-token example takes 5 + 25 / 10 = 7.5 -> 8 tokens of the budget.
        ((5, 5), dict(quadratic_length=10), [[0], [1]]),
        ((5, 5, 2, 2), {}, [[0, 1], [2, 3]]),
        ((5, 5, 2, 2), dict(drop_last=True), [[0, 1]]),
    ],
)
def test_packed_sampler_packs_best_fit_batches(
    cuts_path, kind, num_tokens, kwargs, expected_ids
):
    sampler = make_sampler(kind, cuts_path(num_tokens), **kwargs)

    assert batch_ids(sampler) == expected_ids


@both_samplers
@pytest.mark.parametrize(
    ("num_tokens", "num_output_tokens", "batch_output_tokens", "expected_ids"),
    [
        # The best fit for the token budget pairs the anchor with the last example, which the output cap forbids.
        ((7, 4, 6, 3), (8, 2, 2, 2), None, [[0, 3], [1, 2]]),
        ((7, 4, 6, 3), (8, 2, 2, 2), 9, [[0], [1, 2], [3]]),
        # The exact fit [0, 1] has 12 output tokens; example 1 is dropped and two small examples fill both budgets.
        ((5, 5, 3, 2), (6, 6, 1, 1), 8, [[0, 2, 3], [1]]),
    ],
)
def test_packed_sampler_output_token_cap(
    cuts_path, kind, num_tokens, num_output_tokens, batch_output_tokens, expected_ids
):
    sampler = make_sampler(
        kind,
        cuts_path(num_tokens, num_output_tokens),
        batch_output_tokens=batch_output_tokens,
    )

    assert batch_ids(sampler) == expected_ids


@both_samplers
@pytest.mark.parametrize(
    ("num_tokens", "num_output_tokens", "batch_output_tokens", "match"),
    [
        ((11,), None, None, "individual example.*exceeds batch_tokens=10"),
        ((4, 4), (12, 2), 10, "output tokens.*exceed batch_output_tokens=10"),
        ((2.5,), None, None, "positive integer token lengths"),
    ],
)
def test_packed_sampler_rejects_examples_over_limits(
    cuts_path, kind, num_tokens, num_output_tokens, batch_output_tokens, match
):
    sampler = make_sampler(
        kind,
        cuts_path(num_tokens, num_output_tokens),
        batch_output_tokens=batch_output_tokens,
    )

    with pytest.raises(ValueError, match=match):
        next(iter(sampler))


@pytest.mark.parametrize(
    ("sampler_cls", "kwargs"),
    [
        (PackedSequenceDynamicCutSampler, {}),
        (PackedSequenceDynamicBucketingSampler, dict(duration_bins=[5])),
    ],
)
def test_packed_sampler_rejects_invalid_setup(sampler_cls, kwargs):
    cuts = DummyManifest(CutSet, begin_id=0, end_id=2)
    constraint = PackedTokenConstraint(batch_tokens=10)
    with pytest.raises(ValueError, match="packing_buffer_size must be positive"):
        sampler_cls(cuts, constraint=constraint, packing_buffer_size=0, **kwargs)
    with pytest.raises(ValueError, match="require constraint=PackedTokenConstraint"):
        sampler_cls(cuts, max_duration=10, **kwargs)


@both_samplers
def test_packed_sampler_tuple_inputs(cuts_path, kind):
    first = cuts_path(name="first.jsonl")
    second = cuts_path((1, 1, 1, 1), name="second.jsonl")

    sampler = make_sampler(kind, first, second)

    assert tuple_batch_ids(sampler) == [([0, 3], [0, 3]), ([1, 2], [1, 2])]


@both_samplers
@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("num_inputs", [1, 2])
def test_packed_sampler_resume_preserves_packing_buffer(
    cuts_path, kind, indexed, num_inputs
):
    paths = [cuts_path(name=f"cuts{idx}.jsonl") for idx in range(num_inputs)]
    sampler = make_sampler(kind, *paths, indexed=indexed)
    next(iter(sampler))
    state = sampler.state_dict()
    assert state["packing_buffer_size"] == 4
    if kind == "cut":
        pool_tokens = [(1,) * num_inputs, (2,) * num_inputs]
        assert state["packing_buffer_tokens"] == (pool_tokens if indexed else None)
    elif indexed:
        pool_tokens = [[1] * num_inputs, [2] * num_inputs]
        assert state["bucketer_state"]["bucket_tokens"][0] == pool_tokens

    restored = make_sampler(kind, *paths, indexed=indexed, packing_buffer_size=1)
    restored.load_state_dict(deepcopy(state))

    expected = [[1, 2]] if num_inputs == 1 else [([1, 2], [1, 2])]
    actual = batch_ids(restored) if num_inputs == 1 else tuple_batch_ids(restored)
    remaining = remaining_batches(sampler)
    assert actual == expected
    assert (
        batch_ids(remaining) if num_inputs == 1 else tuple_batch_ids(remaining)
    ) == expected
    assert restored.packing_buffer_size == 4


def test_packed_sampler_multiworker_stateful_resume(cuts_path):
    StatefulDataLoader = pytest.importorskip(
        "torchdata.stateful_dataloader"
    ).StatefulDataLoader
    path = cuts_path([(7, 4, 6, 3)[idx % 4] for idx in range(160)])

    def make_loader():
        sampler = make_sampler("cut", path, indexed=True, packing_buffer_size=16)
        return StatefulDataLoader(
            IterableDatasetWrapper(IdentityDataset(), sampler),
            batch_size=None,
            num_workers=2,
            snapshot_every_n_steps=1,
        )

    expected = batch_ids(make_loader())
    loader = make_loader()
    batches = iter(loader)
    prefix = batch_ids(next(batches) for _ in range(24))
    state = loader.state_dict()
    restored = make_loader()
    restored.load_state_dict(deepcopy(state))

    assert prefix + batch_ids(restored) == expected


def test_packed_bucketing_sampler_never_mixes_buckets(cuts_path):
    sampler = make_sampler("bucketing", cuts_path((2, 6) * 4), duration_bins=(4,))

    batches = batch_ids(sampler)

    assert sorted(idx for batch in batches for idx in batch) == list(range(8))
    assert all(len({idx % 2 for idx in batch}) == 1 for batch in batches)


def test_packed_bucketing_sampler_max_examples_per_bucket(cuts_path):
    sampler = make_sampler(
        "bucketing",
        cuts_path((2, 2, 2, 6, 2, 2)),
        duration_bins=(5, 10),
        bucket_bins=(5, 10),
        max_examples_per_bucket=(2, 1),
    )

    assert sorted(map(sorted, batch_ids(sampler))) == [[0, 1], [2, 4], [3], [5]]


def test_packed_bucketing_sampler_shuffle_is_deterministic(cuts_path):
    num_tokens = (6, 5, 4, 3, 2, 4, 3, 2)
    path = cuts_path(num_tokens)

    batches = batch_ids(make_sampler("bucketing", path, shuffle=True))

    assert batches == batch_ids(make_sampler("bucketing", path, shuffle=True))
    assert sorted(idx for batch in batches for idx in batch) == list(range(8))
    assert all(sum(num_tokens[idx] for idx in batch) <= 10 for batch in batches)


def test_packed_bucketing_sampler_drop_last_does_not_stop_at_stuck_anchor(
    cuts_path,
):
    # Nothing fits next to the 3-token anchor, so its batch is never full;
    # the 8-token batches behind it must still be sampled.
    sampler = make_sampler(
        "bucketing", cuts_path((3,) + (8,) * 40), buffer_size=20, drop_last=True
    )

    assert batch_ids(sampler) == [[idx] for idx in range(41)]
