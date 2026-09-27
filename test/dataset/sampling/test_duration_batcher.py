import json
from collections import Counter
from copy import deepcopy

import pytest

from lhotse import CutSet
from lhotse.dataset.iterable_dataset import IdentityDataset, IterableDatasetWrapper
from lhotse.dataset.sampling.base import TimeConstraint
from lhotse.dataset.sampling.data_source import DataSource
from lhotse.dataset.sampling.dynamic import DurationBatcher, DynamicCutSampler
from lhotse.dataset.sampling.dynamic_bucketing import DynamicBucketingSampler
from lhotse.testing.dummies import dummy_cut


def make_cuts(durations):
    return CutSet.from_cuts(
        dummy_cut(i, duration=duration) for i, duration in enumerate(durations)
    )


def batch_ids(batches):
    return [list(batch.ids) for batch in batches]


def remaining_batches(sampler):
    # Calling iter(sampler) here would intentionally restart the epoch.
    batches = []
    while True:
        try:
            batches.append(next(sampler))
        except StopIteration:
            return batches


@pytest.mark.parametrize("drop_last", [False, True])
@pytest.mark.parametrize("paired", [False, True])
def test_duration_batcher_defers_overflow(drop_last, paired):
    cuts = make_cuts([1] * 5 + [20, 1, 1])
    source = list(zip(cuts, cuts)) if paired else cuts
    batches = list(DurationBatcher(source, max_duration=50, drop_last=drop_last))
    if paired:
        assert all(list(left.ids) == list(right.ids) for left, right in batches)
        batches = [left for left, _ in batches]
    expected = [list(cuts.ids)[:5], list(cuts.ids)[5:7]]
    if not drop_last:
        expected.append(list(cuts.ids)[7:])
    assert batch_ids(batches) == expected
    assert all(len(batch) * max(c.duration for c in batch) <= 50 for batch in batches)


@pytest.mark.parametrize("close_first", [False, True])
def test_duration_batcher_restart_discards_deferred_cut(close_first):
    cuts = make_cuts([1] * 5 + [20, 1, 1])
    batcher = DurationBatcher(cuts, max_duration=50)
    iterator = iter(batcher)
    next(iterator)
    if close_first:
        iterator.close()
    iterator = iter(batcher)
    assert [c.id for batch in iterator for c in batch] == list(cuts.ids)


@pytest.mark.parametrize("close_first", [False, True])
@pytest.mark.parametrize("paired", [False, True])
def test_duration_batcher_reiterating_one_shot_source_keeps_deferred_cut(
    close_first, paired
):
    cuts = make_cuts([1] * 5 + [20, 1, 1])
    source = zip(cuts, cuts) if paired else iter(cuts)
    batcher = DurationBatcher(source, max_duration=50)
    iterator = iter(batcher)
    next(iterator)
    if close_first:
        iterator.close()
    iterator = iter(batcher)
    batches = list(iterator)
    if paired:
        assert all(list(left.ids) == list(right.ids) for left, right in batches)
        batches = [left for left, _ in batches]
    assert batch_ids(batches) == [list(cuts.ids)[5:7], list(cuts.ids)[7:]]


@pytest.mark.parametrize(
    "restart,close_first", [(False, False), (True, False), (True, True)]
)
def test_duration_batcher_resettable_data_source(restart, close_first):
    cuts = make_cuts([1] * 5 + [20, 1, 1])
    source = DataSource(cuts)
    batcher = DurationBatcher(source, max_duration=50)
    iterator = iter(batcher)
    first = next(iterator)
    if restart:
        if close_first:
            iterator.close()
        iterator = iter(batcher)
        sampled = [c.id for batch in iterator for c in batch]
    else:
        sampled = list(first.ids) + [c.id for batch in iterator for c in batch]
    assert sampled == list(cuts.ids)


def test_duration_batcher_oversized_singleton_warns_and_preserves_cuts():
    cuts = make_cuts([1, 60, 1])
    with pytest.warns(UserWarning, match="only 1 cut"):
        batches = list(DurationBatcher(cuts, max_duration=50))
    assert batch_ids(batches) == [[cut.id] for cut in cuts]


@pytest.mark.parametrize(
    "durations,constraint,expected_sizes",
    [
        ([10] * 6, TimeConstraint(max_duration=50), [5, 1]),
        ([1] * 5 + [20, 1], TimeConstraint(max_duration=50, max_cuts=3), [3, 2, 2]),
        (
            [1] * 5 + [20, 1],
            TimeConstraint(max_duration=50, quadratic_duration=20),
            [5, 1, 1],
        ),
        (
            [10, 10, 40, 10],
            TimeConstraint(max_duration=50, concatenate_cuts=True),
            [2, 2],
        ),
    ],
)
def test_duration_batcher_constraints(durations, constraint, expected_sizes):
    cuts = make_cuts(durations)
    batches = list(DurationBatcher(cuts, constraint=constraint))
    assert [len(batch) for batch in batches] == expected_sizes
    assert [c.id for batch in batches for c in batch] == list(cuts.ids)
    for batch in batches:
        constraint.reset()
        for cut in batch:
            constraint.add(cut)
        assert not constraint.exceeded()


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("concurrent", [False, True])
def test_shuffled_buckets_preserve_overflow_cut(seed, concurrent):
    cuts = make_cuts([1] * 25 + [20] * 5)
    sampler = DynamicBucketingSampler(
        cuts,
        max_duration=50,
        duration_bins=[],
        buffer_size=30,
        shuffle=True,
        seed=seed,
        concurrent=concurrent,
    )
    batches = list(sampler)
    assert all(len(batch) * max(c.duration for c in batch) <= 50 for batch in batches)
    assert Counter(c.id for batch in batches for c in batch) == Counter(cuts.ids)


@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("sampler_cls", [DynamicCutSampler, DynamicBucketingSampler])
def test_overflow_checkpoint_resume(
    tmp_path, monkeypatch, indexed, shuffle, paired, sampler_cls
):
    path = tmp_path / "cuts.jsonl"
    make_cuts([1] * 25 + [20] * 5).to_jsonl(path)

    def make_sampler():
        sources = [
            CutSet.from_file(path, indexed=indexed) for _ in range(2 if paired else 1)
        ]
        kwargs = dict(max_duration=50, shuffle=shuffle, seed=0)
        if sampler_cls is DynamicBucketingSampler:
            kwargs.update(duration_bins=[], buffer_size=30)
        else:
            kwargs.update(shuffle_buffer_size=30)
        return sampler_cls(*sources, **kwargs)

    def ids(batches):
        if paired:
            return [(list(left.ids), list(right.ids)) for left, right in batches]
        return batch_ids(batches)

    reference = make_sampler()
    iter(reference)
    next(reference)
    state = deepcopy(reference.state_dict())
    if "batcher_state" in state:
        state["batcher_state"] = json.loads(json.dumps(state["batcher_state"]))
    expected = ids(remaining_batches(reference))
    restored = make_sampler()
    restored.load_state_dict(deepcopy(state))
    if indexed:

        def forbid_replay():
            pytest.fail("Indexed checkpoint restore must not replay batches")

        monkeypatch.setattr(restored, "_replay_step", forbid_replay)
    assert ids(list(restored)) == expected


@pytest.mark.parametrize("advance_restored", [False, True])
def test_deferred_checkpoint_can_be_saved_again_before_next(tmp_path, advance_restored):
    path = tmp_path / "cuts.jsonl"
    cuts = make_cuts([1] * 5 + [20, 1, 1])
    cuts.to_jsonl(path)

    def make_sampler():
        return DynamicCutSampler(CutSet.from_file(path, indexed=True), max_duration=50)

    sampler = make_sampler()
    iter(sampler)
    next(sampler)
    restored = make_sampler()
    restored.load_state_dict(deepcopy(sampler.state_dict()))
    if advance_restored:
        iter(restored)
    restored_again = make_sampler()
    restored_again.load_state_dict(deepcopy(restored.state_dict()))
    assert batch_ids(list(restored_again)) == [list(cuts.ids)[5:7], list(cuts.ids)[7:]]


def test_discard_restored_deferred_cut_when_restarting_epoch(tmp_path):
    path = tmp_path / "cuts.jsonl"
    cuts = make_cuts([1] * 5 + [20, 1, 1])
    cuts.to_jsonl(path)
    sampler = DynamicCutSampler(CutSet.from_file(path, indexed=True), max_duration=50)
    iter(sampler)
    next(sampler)
    sampler.load_state_dict(deepcopy(sampler.state_dict()))
    sampler.allow_iter_to_reset_state()
    sampler.set_epoch(1)
    assert [c.id for batch in sampler for c in batch] == list(cuts.ids)


def resample_cut(cut):
    return cut.resample(8000)


@pytest.mark.parametrize("shuffle", [False, True])
def test_deferred_cut_checkpoint_with_nested_graph_and_filter(tmp_path, shuffle):
    path = tmp_path / "cuts.jsonl"
    cuts = make_cuts([1] * 5 + [20, 1, 1])
    cuts.to_jsonl(path)

    def make_sampler():
        source = (
            CutSet.from_file(path, indexed=True)
            .repeat(times=2, preserve_id=False)
            .map(resample_cut)
        )
        sampler = DynamicCutSampler(
            source, max_duration=50, shuffle=shuffle, shuffle_buffer_size=8, seed=0
        )
        return sampler.filter(lambda cut: "0006" not in cut.id)

    sampler = make_sampler()
    iter(sampler)
    next(sampler)
    state = deepcopy(sampler.state_dict())
    assert state["batcher_state"]  # Checkpoint actually contains a deferred cut.
    state["batcher_state"] = json.loads(json.dumps(state["batcher_state"]))
    expected = [[c.to_dict() for c in b] for b in remaining_batches(sampler)]
    restored = make_sampler()
    restored.load_state_dict(state)
    assert [[c.to_dict() for c in b] for b in restored] == expected


@pytest.mark.parametrize("num_workers", [0, 2])
def test_overflow_stateful_dataloader_resume(tmp_path, num_workers):
    StatefulDataLoader = pytest.importorskip(
        "torchdata.stateful_dataloader"
    ).StatefulDataLoader
    path = tmp_path / "cuts.jsonl"
    make_cuts(([1] * 5 + [20, 1, 1]) * 20).to_jsonl(path)

    def make_loader():
        source = CutSet.from_file(path, indexed=True)
        sampler = DynamicCutSampler(source, max_duration=50)
        dataset = IterableDatasetWrapper(IdentityDataset(), sampler)
        return StatefulDataLoader(dataset, batch_size=None, num_workers=num_workers)

    expected = batch_ids(make_loader())
    loader = make_loader()
    iterator = iter(loader)
    before = batch_ids([next(iterator) for _ in range(3)])
    state = deepcopy(loader.state_dict())
    restored = make_loader()
    restored.load_state_dict(state)
    assert before + batch_ids(restored) == expected


def test_duration_batcher_token_constraint():
    import numpy as np

    from lhotse.cut.text import TextExample
    from lhotse.dataset.sampling.base import TokenConstraint

    examples = [
        TextExample(str(i), np.zeros(n, dtype=np.int32))
        for i, n in enumerate([1] * 5 + [20, 1, 1])
    ]
    batches = list(DurationBatcher(examples, constraint=TokenConstraint(max_tokens=50)))
    assert [len(batch) for batch in batches] == [5, 2, 1]
    assert [e.text for batch in batches for e in batch] == [e.text for e in examples]


def test_dynamic_cut_sampler_loads_checkpoint_without_batcher_state(tmp_path):
    path = tmp_path / "cuts.jsonl"
    cuts = make_cuts([10] * 6)
    cuts.to_jsonl(path)
    sampler = DynamicCutSampler(CutSet.from_file(path, indexed=True), max_duration=50)
    iter(sampler)
    next(sampler)
    state = deepcopy(sampler.state_dict())
    state.pop("batcher_state")
    restored = DynamicCutSampler(CutSet.from_file(path, indexed=True), max_duration=50)
    restored.load_state_dict(state)
    assert batch_ids(restored) == [list(cuts.ids)[5:]]
