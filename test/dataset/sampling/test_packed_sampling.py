from copy import deepcopy
from types import SimpleNamespace

import pytest

from lhotse import CutSet
from lhotse.dataset import (
    PackedSequenceDynamicBucketingSampler,
    PackedSequenceDynamicCutSampler,
    PackedTokenConstraint,
)
from lhotse.dataset.iterable_dataset import IdentityDataset, IterableDatasetWrapper
from lhotse.dataset.sampling.packed import _fit_output_cap, _select_best_fit_indices
from lhotse.indexing import create_jsonl_index
from lhotse.lazy import LazyIndexedManifestIterator
from lhotse.testing.dummies import DummyManifest, dummy_cut


def make_cuts(num_tokens=(7, 4, 6, 3), num_output_tokens=None):
    cuts = []
    for idx, length in enumerate(num_tokens):
        cut = dummy_cut(idx, duration=float(length))
        cut.custom = {"num_tokens": length}
        if num_output_tokens is not None:
            cut.custom["num_output_tokens"] = num_output_tokens[idx]
        cuts.append(cut)
    return CutSet.from_cuts(cuts)


@pytest.fixture
def cuts_path(tmp_path):
    def write(*args, indexed=False, name="cuts.jsonl"):
        path = tmp_path / name
        make_cuts(*args).to_file(path)
        if indexed:
            create_jsonl_index(path)
        return path

    return write


def open_cuts(path, indexed=False):
    if indexed:
        return CutSet(LazyIndexedManifestIterator(path))
    return CutSet.from_jsonl_lazy(path)


def make_sampler(
    *cuts,
    batch_tokens=10,
    max_examples=None,
    quadratic_length=None,
    batch_output_tokens=None,
    packing_buffer_size=4,
    drop_last=False,
):
    return PackedSequenceDynamicCutSampler(
        *cuts,
        constraint=PackedTokenConstraint(
            batch_tokens=batch_tokens,
            max_examples=max_examples,
            quadratic_length=quadratic_length,
            batch_output_tokens=batch_output_tokens,
        ),
        packing_buffer_size=packing_buffer_size,
        drop_last=drop_last,
    )


def make_bucketing_sampler(
    *cuts,
    batch_tokens=10,
    max_examples=None,
    batch_output_tokens=None,
    duration_bins=(100,),
    max_examples_per_bucket=None,
    packing_buffer_size=4,
    buffer_size=32,
    shuffle=False,
    drop_last=False,
    seed=0,
):
    return PackedSequenceDynamicBucketingSampler(
        *cuts,
        constraint=PackedTokenConstraint(
            batch_tokens=batch_tokens,
            max_examples=max_examples,
            batch_output_tokens=batch_output_tokens,
            bucket_bins=(
                list(duration_bins) if max_examples_per_bucket is not None else None
            ),
            max_examples_per_bucket=max_examples_per_bucket,
        ),
        duration_bins=list(duration_bins),
        packing_buffer_size=packing_buffer_size,
        buffer_size=buffer_size,
        shuffle=shuffle,
        drop_last=drop_last,
        seed=seed,
    )


def batch_ids(batches):
    return [[int(cut.id[-4:]) for cut in batch] for batch in batches]


def remaining_batch_ids(sampler):
    """Drain the current epoch; ``iter(sampler)`` would start a new one."""
    batches = []
    while True:
        try:
            batches.append(next(sampler))
        except StopIteration:
            return batch_ids(batches)


def test_select_best_fit_indices_is_exact_and_prefers_earlier_items():
    assert _select_best_fit_indices([4, 6, 3, 3, 1], capacity=7) == [0, 2]
    assert _select_best_fit_indices([4, 6, 3, 3, 1], capacity=7, max_items=1) == [1]
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


def test_packed_sampler_backfills_up_to_exact_token_cap(cuts_path):
    batches = list(make_sampler(open_cuts(cuts_path())))

    # A greedy batcher stops before the 4/6-token examples; best-fit packing
    # reaches past them for the 3-token example.
    assert batch_ids(batches) == [[0, 3], [1, 2]]
    assert [sum(cut.num_tokens for cut in batch) for batch in batches] == [10, 10]


def test_packed_sampler_squeezes_around_large_next_candidate(cuts_path):
    batches = list(make_sampler(open_cuts(cuts_path((9, 12, 4, 3))), batch_tokens=16))

    assert batch_ids(batches) == [[0, 2, 3], [1]]


def test_packed_sampler_anchor_prevents_starvation(cuts_path):
    batches = list(make_sampler(open_cuts(cuts_path((6, 6, 4, 4, 4, 4)))))

    # Example 1 doesn't fit beside example 0, but it anchors the very next batch.
    assert batch_ids(batches) == [[0, 2], [1, 3], [4, 5]]


def test_packed_sampler_max_examples(cuts_path):
    batches = list(make_sampler(open_cuts(cuts_path((2, 2, 2, 2))), max_examples=2))

    assert batch_ids(batches) == [[0, 1], [2, 3]]


def test_packed_sampler_quadratic_length(cuts_path):
    # Each 5-token example takes 5 + 25 / 10 = 7.5 -> 8 tokens of the budget.
    batches = list(make_sampler(open_cuts(cuts_path((5, 5))), quadratic_length=10))

    assert batch_ids(batches) == [[0], [1]]


@pytest.mark.parametrize(
    ("drop_last", "expected_ids"), [(False, [[0, 1], [2, 3]]), (True, [[0, 1]])]
)
def test_packed_sampler_drop_last(cuts_path, drop_last, expected_ids):
    batches = list(
        make_sampler(open_cuts(cuts_path((5, 5, 2, 2))), drop_last=drop_last)
    )

    assert batch_ids(batches) == expected_ids


def test_packed_sampler_output_token_cap(cuts_path):
    path = cuts_path((7, 4, 6, 3), (8, 2, 2, 2))

    # The best fit for the token budget pairs the anchor with the last example, which the output cap forbids.
    assert batch_ids(make_sampler(open_cuts(path))) == [[0, 3], [1, 2]]
    assert batch_ids(make_sampler(open_cuts(path), batch_output_tokens=9)) == [
        [0],
        [1, 2],
        [3],
    ]


def test_packed_sampler_backfills_within_output_cap(cuts_path):
    path = cuts_path((5, 5, 3, 2), (6, 6, 1, 1))

    batches = list(make_sampler(open_cuts(path), batch_output_tokens=8))

    # The exact fit [0, 1] has 12 output tokens; example 1 is dropped and two small examples fill both budgets.
    assert batch_ids(batches) == [[0, 2, 3], [1]]


@pytest.mark.parametrize(
    ("num_tokens", "num_output_tokens", "batch_output_tokens", "match"),
    [
        ((11,), None, None, "individual example.*exceeds batch_tokens=10"),
        ((4, 4), (12, 2), 10, "output tokens.*exceed batch_output_tokens=10"),
        ((2.5,), None, None, "positive integer token lengths"),
    ],
)
def test_packed_sampler_rejects_examples_over_limits(
    cuts_path, num_tokens, num_output_tokens, batch_output_tokens, match
):
    sampler = make_sampler(
        open_cuts(cuts_path(num_tokens, num_output_tokens)),
        batch_output_tokens=batch_output_tokens,
    )

    with pytest.raises(ValueError, match=match):
        next(iter(sampler))


def test_packed_sampler_rejects_invalid_setup():
    cuts = DummyManifest(CutSet, begin_id=0, end_id=2)
    with pytest.raises(ValueError, match="packing_buffer_size must be positive"):
        make_sampler(cuts, packing_buffer_size=0)
    with pytest.raises(ValueError, match="require constraint=PackedTokenConstraint"):
        PackedSequenceDynamicCutSampler(cuts, max_duration=10)


def test_packed_sampler_tuple_inputs(cuts_path):
    first = open_cuts(cuts_path(name="first.jsonl"))
    second = open_cuts(cuts_path((1, 1, 1, 1), name="second.jsonl"))

    batches = list(make_sampler(first, second))

    assert [tuple(batch_ids(batch)) for batch in batches] == [
        ([0, 3], [0, 3]),
        ([1, 2], [1, 2]),
    ]


@pytest.mark.parametrize("indexed", [False, True])
def test_packed_sampler_resume_preserves_packing_buffer(cuts_path, indexed):
    path = cuts_path(indexed=indexed)
    sampler = make_sampler(open_cuts(path, indexed))
    assert batch_ids([next(iter(sampler))]) == [[0, 3]]
    state = sampler.state_dict()
    assert state["packing_buffer_size"] == 4
    assert state["packing_buffer_tokens"] == ([(1,), (2,)] if indexed else None)

    restored = make_sampler(open_cuts(path, indexed), packing_buffer_size=1)
    restored.load_state_dict(deepcopy(state))

    assert batch_ids(restored) == remaining_batch_ids(sampler) == [[1, 2]]
    assert restored.packing_buffer_size == 4


def test_packed_sampler_indexed_tuple_inputs_resume(cuts_path):
    paths = (
        cuts_path(name="first.jsonl", indexed=True),
        cuts_path((1, 1, 1, 1), name="second.jsonl", indexed=True),
    )
    sampler = make_sampler(*(open_cuts(p, indexed=True) for p in paths))
    next(iter(sampler))
    state = sampler.state_dict()
    assert state["packing_buffer_tokens"] == [(1, 1), (2, 2)]

    restored = make_sampler(*(open_cuts(p, indexed=True) for p in paths))
    restored.load_state_dict(deepcopy(state))

    assert [tuple(batch_ids(b)) for b in restored] == [([1, 2], [1, 2])]
    assert [tuple(batch_ids(b)) for b in sampler] == [
        ([0, 3], [0, 3]),
        ([1, 2], [1, 2]),
    ]


def test_packed_sampler_multiworker_stateful_resume(tmp_path):
    StatefulDataLoader = pytest.importorskip(
        "torchdata.stateful_dataloader"
    ).StatefulDataLoader
    path = tmp_path / "cuts.jsonl"
    make_cuts([(7, 4, 6, 3)[idx % 4] for idx in range(160)]).to_file(path)
    create_jsonl_index(path)

    def make_loader():
        sampler = make_sampler(open_cuts(path, indexed=True), packing_buffer_size=16)
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


def test_packed_bucketing_sampler_backfills_up_to_exact_token_cap(cuts_path):
    batches = list(make_bucketing_sampler(open_cuts(cuts_path()), buffer_size=4))

    assert batch_ids(batches) == [[0, 3], [1, 2]]


def test_packed_bucketing_sampler_anchor_prevents_starvation(cuts_path):
    batches = list(
        make_bucketing_sampler(open_cuts(cuts_path((6, 6, 4, 4, 4, 4))), buffer_size=6)
    )

    assert batch_ids(batches) == [[0, 2], [1, 3], [4, 5]]


def test_packed_bucketing_sampler_never_mixes_buckets(cuts_path):
    batches = list(
        make_bucketing_sampler(
            open_cuts(cuts_path((2, 6) * 4)), buffer_size=8, duration_bins=(4,)
        )
    )

    assert sorted(idx for batch in batch_ids(batches) for idx in batch) == list(
        range(8)
    )
    for batch in batch_ids(batches):
        assert len({idx % 2 for idx in batch}) == 1


def test_packed_bucketing_sampler_max_examples_per_bucket(cuts_path):
    batches = list(
        make_bucketing_sampler(
            open_cuts(cuts_path((2, 2, 2, 6, 2, 2))),
            buffer_size=6,
            duration_bins=(5, 10),
            max_examples_per_bucket=(2, 1),
        )
    )

    assert sorted(map(sorted, batch_ids(batches))) == [[0, 1], [2, 4], [3], [5]]


def test_packed_bucketing_sampler_output_token_cap(cuts_path):
    path = cuts_path((7, 4, 6, 3), (8, 2, 2, 2))

    batches = list(make_bucketing_sampler(open_cuts(path), batch_output_tokens=9))

    assert batch_ids(batches) == [[0], [1, 2], [3]]


def test_packed_bucketing_sampler_tuple_inputs(cuts_path):
    first = open_cuts(cuts_path(name="first.jsonl"))
    second = open_cuts(cuts_path((1, 1, 1, 1), name="second.jsonl"))

    batches = list(make_bucketing_sampler(first, second, buffer_size=4))

    assert [tuple(batch_ids(batch)) for batch in batches] == [
        ([0, 3], [0, 3]),
        ([1, 2], [1, 2]),
    ]


def test_packed_bucketing_sampler_shuffle_is_deterministic(cuts_path):
    path = cuts_path((6, 5, 4, 3, 2, 4, 3, 2))

    def sample():
        sampler = make_bucketing_sampler(
            open_cuts(path), buffer_size=8, shuffle=True, seed=31
        )
        return batch_ids(sampler)

    batches = sample()
    assert batches == sample()
    assert sorted(idx for batch in batches for idx in batch) == list(range(8))
    assert all(sum((6, 5, 4, 3, 2, 4, 3, 2)[i] for i in b) <= 10 for b in batches)


@pytest.mark.parametrize(
    ("drop_last", "expected_ids"), [(False, [0, 1, 2, 3]), (True, [0, 1])]
)
def test_packed_bucketing_sampler_drop_last(cuts_path, drop_last, expected_ids):
    batches = list(
        make_bucketing_sampler(
            open_cuts(cuts_path((5, 5, 2, 2))), buffer_size=4, drop_last=drop_last
        )
    )

    assert [idx for batch in batch_ids(batches) for idx in batch] == expected_ids


def test_packed_bucketing_sampler_indexed_resume_preserves_arbitrary_removals(
    cuts_path,
):
    path = cuts_path(indexed=True)
    sampler = make_bucketing_sampler(open_cuts(path, indexed=True), buffer_size=4)
    assert batch_ids([next(iter(sampler))]) == [[0, 3]]
    state = sampler.state_dict()
    assert state["packing_buffer_size"] == 4
    assert state["bucketer_state"]["bucket_tokens"][0] == [[1], [2]]

    restored = make_bucketing_sampler(
        open_cuts(path, indexed=True), buffer_size=4, packing_buffer_size=1
    )
    restored.load_state_dict(deepcopy(state))

    assert batch_ids(restored) == remaining_batch_ids(sampler) == [[1, 2]]
    assert restored.packing_buffer_size == 4
