import math
import random
import re
from collections import Counter
from copy import deepcopy
from functools import partial
from math import isclose
from statistics import mean
from tempfile import NamedTemporaryFile

import pytest
from torch.utils.data import DataLoader

from lhotse import CutSet
from lhotse.dataset import (
    CutConcatenate,
    DynamicBucketingSampler,
    IterableDatasetWrapper,
    RoundRobinSampler,
    make_worker_init_fn,
    report_padding_ratio_estimate,
)
from lhotse.dataset.cut_transforms import concat_cuts
from lhotse.dataset.sampling import (
    BucketingSampler,
    CutPairsSampler,
    SimpleCutSampler,
    WeightedSimpleCutSampler,
    ZipSampler,
)
from lhotse.dataset.sampling.base import SamplingDiagnostics, TimeConstraint
from lhotse.dataset.sampling.dynamic import DynamicCutSampler
from lhotse.testing.dummies import DummyManifest, as_lazy, dummy_cut
from lhotse.utils import fastcopy, streaming_shuffle


@pytest.fixture
def libri_cut_set():
    cs = CutSet.from_json("test/fixtures/libri/cuts.json")
    return CutSet.from_cuts(
        [cs[0], cs[0].with_id("copy-1"), cs[0].with_id("copy-2"), cs[0].append(cs[0])]
    )


def test_dynamic_cut_sampler_max_cuts():
    # The dummy cuts have a duration of 1 second each
    cut_set = DummyManifest(CutSet, begin_id=0, end_id=20)

    sampler = DynamicCutSampler(cut_set, max_cuts=5)

    tot = 0
    for batch in sampler:
        assert len(batch) == 5
        tot += 1

    assert tot == 4


def test_dynamic_cut_sampler_quadratic_duration():
    # 2 cuts of 2s followed by 3 cuts of 1s
    cut_set = DummyManifest(CutSet, begin_id=0, end_id=5)
    for i, c in enumerate(cut_set):
        if i < 2:
            c.duration = 2.0

    # at quadratic_duration=2.0, cuts of 1s have 1.5s and cuts of 2s have 4s
    sampler = DynamicCutSampler(cut_set, max_duration=8.0, quadratic_duration=2.0)

    batches = [b for b in sampler]
    assert len(batches) == 2

    b = batches[0]
    assert len(b) == 2
    assert sum(c.duration for c in b) == 4.0

    b = batches[1]
    assert len(b) == 3
    assert sum(c.duration for c in b) == 3.0


@pytest.mark.parametrize("sampler_cls", [SimpleCutSampler, DynamicCutSampler])
def test_single_cut_sampler_shuffling(sampler_cls):
    # The dummy cuts have a duration of 1 second each
    cut_set = DummyManifest(CutSet, begin_id=0, end_id=100)

    sampler = sampler_cls(
        cut_set,
        shuffle=True,
        # Set an effective batch size of 10 cuts, as all have 1s duration == 100 frames
        # This way we're testing that it works okay when returning multiple batches in
        # a full epoch.
        max_duration=10.0,
    )
    sampled_cuts = []
    for batch in sampler:
        sampled_cuts.extend(batch)

    # Invariant 1: we receive the same amount of items in a dataloader epoch as there we in the CutSet
    assert len(sampled_cuts) == len(cut_set)
    # Invariant 2: the items are not duplicated
    assert len(set(c.id for c in sampled_cuts)) == len(sampled_cuts)
    # Invariant 3: the items are shuffled, i.e. the order is different than that in the CutSet
    assert [c.id for c in sampled_cuts] != [c.id for c in cut_set]


class IdentityDataset:
    def __getitem__(self, item):
        return item


@pytest.mark.parametrize("sampler_cls", [DynamicCutSampler, DynamicBucketingSampler])
@pytest.mark.parametrize("seed", [0, "randomized", "trng"])
def test_shuffle_seed_strategies(sampler_cls, seed):
    cut_set = DummyManifest(CutSet, begin_id=0, end_id=100)

    world_size = 2
    sampled_cuts = []
    for rank in range(world_size):
        sampler = sampler_cls(
            cut_set,
            shuffle=True,
            max_duration=10.0,
            seed=seed,
            rank=0,
            world_size=1,
        )
        dloader = DataLoader(
            IterableDatasetWrapper(IdentityDataset(), sampler),
            num_workers=2,
            batch_size=None,
            worker_init_fn=make_worker_init_fn(rank=rank, world_size=world_size),
        )
        for batch in dloader:
            sampled_cuts.extend(batch)

    # Since we're using 2 nodes * 2 workers, an iterable dataset, and do not do anything to de-duplicate,
    # we have 4 copies of the input data.
    assert len(sampled_cuts) == 4 * len(cut_set)
    uniq_ids = Counter()
    for c in sampled_cuts:
        uniq_ids[c.id] += 1
    assert all(v == 4 for v in uniq_ids.values())

    input_ids = list(cut_set.ids)
    node0_worker0 = [
        c.id
        for c in sampled_cuts
        if c.dataloading_info["worker_id"] == 0 and c.dataloading_info["rank"] == 0
    ]
    node0_worker1 = [
        c.id
        for c in sampled_cuts
        if c.dataloading_info["worker_id"] == 1 and c.dataloading_info["rank"] == 0
    ]
    node1_worker0 = [
        c.id
        for c in sampled_cuts
        if c.dataloading_info["worker_id"] == 0 and c.dataloading_info["rank"] == 1
    ]
    node1_worker1 = [
        c.id
        for c in sampled_cuts
        if c.dataloading_info["worker_id"] == 1 and c.dataloading_info["rank"] == 1
    ]

    if seed == 0:
        # When seed=0, ensure each copy is shuffled in the same order (but different than the input).
        assert node0_worker0 == node0_worker1
        assert node0_worker0 == node1_worker0
        assert node0_worker0 == node1_worker1
        assert node0_worker0 != input_ids
    else:
        # Otherwise, we expect each worker to shuffle in a different order.
        assert node0_worker0 != node0_worker1
        assert node0_worker0 != node1_worker0
        assert node0_worker0 != node1_worker1
        assert node0_worker1 != node1_worker0
        assert node0_worker1 != node1_worker1
        assert node1_worker0 != node1_worker1
        assert node0_worker0 != input_ids
        assert node0_worker1 != input_ids
        assert node1_worker0 != input_ids
        assert node1_worker1 != input_ids


def test_single_cut_sampler_time_constraints():
    # The dummy cuts have a duration of 1 second each
    cut_set = DummyManifest(CutSet, begin_id=0, end_id=100)

    sampler = SimpleCutSampler(
        cut_set,
        shuffle=True,
        # Set an effective batch size of 10 cuts, as all have 1s duration == 100 frames
        # This way we're testing that it works okay when returning multiple batches in
        # a full epoch.
        max_duration=10.0,
    )
    sampler_cut_ids = []
    for batch in sampler:
        sampler_cut_ids.extend(batch)

    # Invariant 1: we receive the same amount of items in a dataloader epoch as there we in the CutSet
    assert len(sampler_cut_ids) == len(cut_set)
    # Invariant 2: the items are not duplicated
    assert len(set(c.id for c in sampler_cut_ids)) == len(sampler_cut_ids)
    # Invariant 3: the items are shuffled, i.e. the order is different than that in the CutSet
    assert [c.id for c in sampler_cut_ids] != [c.id for c in cut_set]


@pytest.mark.parametrize("sampler_cls", [SimpleCutSampler, DynamicCutSampler])
def test_single_cut_sampler_order_is_deterministic_given_epoch(sampler_cls):
    cut_set = DummyManifest(CutSet, begin_id=0, end_id=100)

    sampler = sampler_cls(
        cut_set,
        shuffle=True,
        # Set an effective batch size of 10 cuts, as all have 1s duration == 100 frames
        # This way we're testing that it works okay when returning multiple batches in
        # a full epoch.
        max_duration=10.0,
    )
    sampler.set_epoch(42)
    # calling the sampler twice without epoch update gives identical ordering
    assert [item for item in sampler] == [item for item in sampler]


@pytest.mark.parametrize("sampler_cls", [SimpleCutSampler, DynamicCutSampler])
def test_single_cut_sampler_order_differs_between_epochs(sampler_cls):
    cut_set = DummyManifest(CutSet, begin_id=0, end_id=100)

    sampler = sampler_cls(
        cut_set,
        shuffle=True,
        # Set an effective batch size of 10 cuts, as all have 1s duration == 100 frames
        # This way we're testing that it works okay when returning multiple batches in
        # a full epoch.
        max_duration=10.0,
    )
    last_order = [item for item in sampler]
    for epoch in range(1, 6):
        sampler.set_epoch(epoch)
        new_order = [item for item in sampler]
        assert new_order != last_order
        last_order = new_order


@pytest.mark.parametrize("sampler_cls", [SimpleCutSampler, DynamicCutSampler])
def test_single_cut_sampler_low_max_frames(libri_cut_set, sampler_cls):
    sampler = sampler_cls(libri_cut_set, shuffle=False, max_duration=0.02)
    # Check that it does not crash
    for batch in sampler:
        # There will be only a single item in each batch as we're exceeding the limit each time.
        assert len(batch) == 1


def test_cut_pairs_sampler():
    # The dummy cuts have a duration of 1 second each
    cut_set = DummyManifest(CutSet, begin_id=0, end_id=100)

    sampler = CutPairsSampler(
        source_cuts=cut_set,
        target_cuts=cut_set,
        shuffle=True,
        # Set an effective batch size of 10 cuts, as all have 1s duration == 100 frames
        # This way we're testing that it works okay when returning multiple batches in
        # a full epoch.
        max_source_duration=100.0,
        max_target_duration=50.0,
    )
    source_cuts, target_cuts = [], []
    for src_batch, tgt_batch in sampler:
        source_cuts.extend(src_batch)
        target_cuts.extend(tgt_batch)

    # Invariant 1: we receive the same amount of items in a dataloader epoch as there we in the CutSet
    assert len(source_cuts) == len(cut_set)
    assert len(target_cuts) == len(cut_set)
    # Invariant 2: the items are not duplicated
    assert len(set(c.id for c in source_cuts)) == len(source_cuts)
    assert len(set(c.id for c in target_cuts)) == len(target_cuts)
    # Invariant 3: the items are shuffled, i.e. the order is different than that in the CutSet
    assert [c.id for c in source_cuts] != [c.id for c in cut_set]
    # Invariant 4: the source and target cuts are in the same order
    assert [c.id for c in source_cuts] == [c.id for c in target_cuts]


def test_dynamic_cut_sampler_as_cut_pairs_sampler():
    # The dummy cuts have a duration of 1 second each
    cut_set = DummyManifest(CutSet, begin_id=0, end_id=100)

    sampler = DynamicCutSampler(
        cut_set,
        cut_set,
        shuffle=True,
        max_duration=5.0,
    )
    source_cuts, target_cuts = [], []
    for src_batch, tgt_batch in sampler:
        source_cuts.extend(src_batch)
        target_cuts.extend(tgt_batch)

    # Invariant 1: we receive the same amount of items in a dataloader epoch as there we in the CutSet
    assert len(source_cuts) == len(cut_set)
    assert len(target_cuts) == len(cut_set)
    # Invariant 2: the items are not duplicated
    assert len(set(c.id for c in source_cuts)) == len(source_cuts)
    assert len(set(c.id for c in target_cuts)) == len(target_cuts)
    # Invariant 3: the items are shuffled, i.e. the order is different than that in the CutSet
    assert [c.id for c in source_cuts] != [c.id for c in cut_set]
    # Invariant 4: the source and target cuts are in the same order
    assert [c.id for c in source_cuts] == [c.id for c in target_cuts]


def test_cut_pairs_sampler_2():
    cut_set = CutSet.from_cuts(
        [
            dummy_cut(0, duration=10),
            dummy_cut(1, duration=20),
        ]
    )
    sampler = CutPairsSampler(
        source_cuts=cut_set,
        target_cuts=cut_set,
        max_source_duration=50,
        max_target_duration=50,
    )
    batch = next(iter(sampler))
    assert len(batch) == 2


def test_cut_pairs_sampler_order_is_deterministic_given_epoch():
    # The dummy cuts have a duration of 1 second each
    cut_set = DummyManifest(CutSet, begin_id=0, end_id=100)

    sampler = CutPairsSampler(
        source_cuts=cut_set,
        target_cuts=cut_set,
        shuffle=True,
        # Set an effective batch size of 10 cuts, as all have 1s duration == 100 frames
        # This way we're testing that it works okay when returning multiple batches in
        # a full epoch.
        max_source_duration=100,
        max_target_duration=50,
    )
    sampler.set_epoch(42)
    # calling the sampler twice without epoch update gives identical ordering
    assert [item for item in sampler] == [item for item in sampler]


def test_cut_pairs_sampler_order_differs_between_epochs():
    # The dummy cuts have a duration of 1 second each
    cut_set = DummyManifest(CutSet, begin_id=0, end_id=100)

    sampler = CutPairsSampler(
        source_cuts=cut_set,
        target_cuts=cut_set,
        shuffle=True,
        # Set an effective batch size of 10 cuts, as all have 1s duration == 100 frames
        # This way we're testing that it works okay when returning multiple batches in
        # a full epoch.
        max_source_duration=100,
        max_target_duration=50,
    )

    last_order = [item for item in sampler]
    for epoch in range(1, 6):
        sampler.set_epoch(epoch)
        new_order = [item for item in sampler]
        assert new_order != last_order
        last_order = new_order


def test_concat_cuts():
    cuts = [
        dummy_cut(0, duration=30.0),
        dummy_cut(1, duration=20.0),
        dummy_cut(2, duration=10.0),
        dummy_cut(3, duration=5.0),
        dummy_cut(4, duration=4.0),
        dummy_cut(5, duration=3.0),
        dummy_cut(6, duration=2.0),
    ]
    concat = concat_cuts(cuts, gap=1.0)
    assert [c.duration for c in concat] == [
        30.0,
        20.0 + 1.0 + 2.0 + 1.0 + 3.0,  # == 27.0
        10.0 + 1.0 + 4.0 + 1.0 + 5.0,  # == 21.0
    ]


def test_concat_cuts_with_duration_factor():
    cuts = [
        dummy_cut(0, duration=10.0),
        dummy_cut(1, duration=8.0),
        dummy_cut(2, duration=6.0),
        dummy_cut(3, duration=5.0),
        dummy_cut(4, duration=4.0),
        dummy_cut(5, duration=3.0),
        dummy_cut(6, duration=2.0),
    ]
    concat = concat_cuts(cuts, gap=1.0, max_duration=20.0)
    assert [c.duration for c in concat] == [
        10.0 + 1.0 + 2.0 + 1.0 + 3.0,  # == 17.0
        8.0 + 1.0 + 4.0 + 1.0 + 5.0,  # == 19.0
        6.0,  # == 6.0
    ]


def test_bucketing_sampler_single_cuts():
    cut_set = DummyManifest(CutSet, begin_id=0, end_id=1000)
    sampler = BucketingSampler(cut_set, sampler_type=SimpleCutSampler, max_cuts=10000)
    sampled_cuts = []
    for batch in sampler:
        sampled_cuts.extend(batch)
    assert set(cut_set.ids) == set(c.id for c in sampled_cuts)


def test_bucketing_sampler_no_issue_with_first_bucket_index_being_minus_one():
    rng = random.Random(42)
    cut_set = DummyManifest(CutSet, begin_id=0, end_id=11)
    for c in cut_set:
        c.duration = rng.randint(1, 10)
    sampler = BucketingSampler(cut_set, num_buckets=2, max_cuts=10000)
    for batch in sampler:
        pass  # does not raise


def test_bucketing_sampler_single_cuts_equal_duration():
    cut_set = DummyManifest(CutSet, begin_id=0, end_id=1000)
    for idx, c in enumerate(cut_set):
        c.duration = (
            3 + idx * 1 / 50
        )  # each cut has a different duration between [3, 23]
    sampler = BucketingSampler(
        cut_set, sampler_type=SimpleCutSampler, num_buckets=10, max_cuts=10000
    )

    # Ensure that each consecutive bucket has less cuts than the previous one
    sampled_cuts, bucket_cum_durs = [], []
    prev_min, prev_max = 0, 0
    num_overlapping_bins = 0
    for (bucket,) in sampler.buckets:
        bucket_durs = [c.duration for c in bucket]
        sampled_cuts.extend(c for c in bucket)
        bucket_cum_durs.append(sum(bucket_durs))
        bucket_min, bucket_max = min(bucket_durs), max(bucket_durs)
        # Ensure that bucket lengths do not overlap, except for the middle
        # 3 buckets maybe
        if prev_max > bucket_min:
            num_overlapping_bins += 1
        assert num_overlapping_bins < 3
        prev_min = bucket_min
        prev_max = bucket_max

    # Assert that all bucket cumulative durations are within 1/10th of the mean
    mean_bucket_dur = mean(bucket_cum_durs)  # ~ 1300s
    for d in bucket_cum_durs:
        assert abs(d - mean_bucket_dur) < 0.1 * mean_bucket_dur

    assert set(cut_set.ids) == set(c.id for c in sampled_cuts)


def test_bucketing_sampler_shuffle():
    cut_set = DummyManifest(CutSet, begin_id=0, end_id=10)
    sampler = BucketingSampler(
        cut_set,
        sampler_type=SimpleCutSampler,
        shuffle=True,
        num_buckets=2,
        max_duration=20.0,
    )

    sampler.set_epoch(0)
    batches_ep0 = []
    for batch in sampler:
        # Convert List[str] to Tuple[str, ...] so that it's hashable
        batches_ep0.append(tuple(c.id for c in batch))
    assert set(cut_set.ids) == set(cid for batch in batches_ep0 for cid in batch)

    sampler.set_epoch(1)
    batches_ep1 = []
    for batch in sampler:
        batches_ep1.append(tuple(c.id for c in batch))
    assert set(cut_set.ids) == set(cid for batch in batches_ep1 for cid in batch)

    # BucketingSampler ordering may be different in different epochs (=> use set() to make it irrelevant)
    # Internal sampler (SimpleCutSampler) ordering should be different in different epochs
    assert set(batches_ep0) != set(batches_ep1)


def test_bucketing_sampler_cut_pairs():
    cut_set1 = DummyManifest(CutSet, begin_id=0, end_id=1000)
    cut_set2 = DummyManifest(CutSet, begin_id=0, end_id=1000)
    sampler = BucketingSampler(cut_set1, cut_set2, sampler_type=CutPairsSampler)

    src_cuts, tgt_cuts = [], []
    for src_batch, tgt_batch in sampler:
        src_cuts.extend(src_batch)
        tgt_cuts.extend(tgt_batch)
    assert set(cut_set1.ids) == set(c.id for c in src_cuts)
    assert set(cut_set2.ids) == set(c.id for c in tgt_cuts)


@pytest.mark.parametrize("shuffle", [False, True])
def test_bucketing_sampler_cut_pairs_equal_duration(shuffle):
    cut_set = DummyManifest(CutSet, begin_id=0, end_id=1000)
    for idx, c in enumerate(cut_set):
        c.duration = (
            3 + idx * 1 / 50
        )  # each cut has a different duration between [3, 23]
    # Target CutSet is going to have different durations
    # -- make sure the bucketing works well with that.
    cut_set_tgt = cut_set.map(lambda c: fastcopy(c, duration=1 / c.duration))

    sampler = BucketingSampler(
        cut_set,
        cut_set_tgt,
        sampler_type=CutPairsSampler,
        num_buckets=10,
        shuffle=shuffle,
    )

    # Ensure that each consecutive bucket has less cuts than the previous one
    prev_len = float("inf")
    bucket_cum_durs = []
    for bucket_src, bucket_tgt in sampler.buckets:
        assert list(bucket_src.ids) == list(bucket_tgt.ids)
        bucket_cum_durs.append(sum(c.duration for c in bucket_src))
        curr_len = len(bucket_src)
        assert curr_len < prev_len
        prev_len = curr_len

    # Assert that all bucket cumulative durations are within 1/10th of the mean
    mean_bucket_dur = mean(bucket_cum_durs)  # ~ 1300s
    for d in bucket_cum_durs:
        assert abs(d - mean_bucket_dur) < 0.1 * mean_bucket_dur


def test_bucketing_sampler_order_is_deterministic_given_epoch():
    cut_set = DummyManifest(CutSet, begin_id=0, end_id=1000)
    sampler = BucketingSampler(cut_set, sampler_type=SimpleCutSampler, max_cuts=10000)

    sampler.set_epoch(42)
    # calling the sampler twice without epoch update gives identical ordering
    assert [item for item in sampler] == [item for item in sampler]


def test_bucketing_sampler_order_differs_between_epochs():
    cut_set = DummyManifest(CutSet, begin_id=0, end_id=1000)
    sampler = BucketingSampler(cut_set, sampler_type=SimpleCutSampler, max_cuts=10000)

    last_order = [item for item in sampler]
    for epoch in range(1, 6):
        sampler.set_epoch(epoch)
        new_order = [item for item in sampler]
        assert new_order != last_order
        last_order = new_order


@pytest.mark.parametrize("world_size", [2, 3, 4])
@pytest.mark.parametrize("n_cuts", [995, 996, 997, 998, 999, 1000, 1001, 1002, 1003])
@pytest.mark.parametrize(
    "sampler_cls", [SimpleCutSampler, BucketingSampler, DynamicCutSampler]
)
def test_partitions_are_equal(world_size, n_cuts, sampler_cls):
    # Create a dummy CutSet.
    cut_set = DummyManifest(CutSet, begin_id=0, end_id=n_cuts)
    # Randomize the durations of cuts to increase the chance we run into edge cases.
    for c in cut_set:
        c.duration += 10 * random.random()
    # Create a sampler for each "distributed worker."
    samplers = [
        sampler_cls(
            cut_set, max_duration=25.0, drop_last=True, rank=i, world_size=world_size
        )
        for i in range(world_size)
    ]
    # Check that it worked.
    n_batches = [len([b for b in s]) for s in samplers]
    assert all(nb == n_batches[0] for nb in n_batches)


def test_bucketing_sampler_raises_value_error_on_lazy_cuts_input():
    cut_set = DummyManifest(CutSet, begin_id=0, end_id=2)
    with NamedTemporaryFile(suffix=".jsonl") as f:
        cut_set.to_jsonl(f.name)
        lazy_cuts = CutSet.from_jsonl_lazy(f.name)
        with pytest.raises(ValueError):
            sampler = BucketingSampler(
                lazy_cuts,
                max_duration=10.0,
            )


@pytest.mark.parametrize(
    "sampler_cls",
    [
        SimpleCutSampler,
        DynamicCutSampler,
    ],
)
def test_single_cut_sampler_with_lazy_cuts(sampler_cls):
    # The dummy cuts have a duration of 1 second each
    cut_set = DummyManifest(CutSet, begin_id=0, end_id=100)
    with NamedTemporaryFile(suffix=".jsonl") as f:
        cut_set.to_jsonl(f.name)
        lazy_cuts = CutSet.from_jsonl_lazy(f.name)

        sampler = sampler_cls(
            lazy_cuts,
            shuffle=False,
            # Set an effective batch size of 10 cuts, as all have 1s duration == 100 frames
            # This way we're testing that it works okay when returning multiple batches in
            # a full epoch.
            max_duration=10.0,
        )
        sampled_cuts = []
        for batch in sampler:
            sampled_cuts.extend(batch)

        # Invariant 1: we receive the same amount of items in a dataloader epoch as there we in the CutSet
        assert len(sampled_cuts) == len(cut_set)
        # Invariant 2: the items are not duplicated
        assert len(set(c.id for c in sampled_cuts)) == len(sampled_cuts)


@pytest.mark.parametrize(
    "sampler_cls",
    [
        SimpleCutSampler,
        DynamicCutSampler,
    ],
)
def test_single_cut_sampler_with_lazy_cuts_concat(sampler_cls):
    # The dummy cuts have a duration of 1 second each
    eager1 = DummyManifest(CutSet, begin_id=0, end_id=100)
    eager2 = DummyManifest(CutSet, begin_id=1000, end_id=1100)
    eager_cuts = eager1 + eager2
    with as_lazy(eager1) as lazy1, as_lazy(eager2) as lazy2:
        lazy_cuts = lazy1 + lazy2

        sampler = sampler_cls(
            lazy_cuts,
            shuffle=False,
            # Set an effective batch size of 10 cuts, as all have 1s duration == 100 frames
            # This way we're testing that it works okay when returning multiple batches in
            # a full epoch.
            max_duration=10.0,
        )
        sampled_cuts = []
        for batch in sampler:
            sampled_cuts.extend(batch)

        # Invariant 1: we receive the same amount of items in a dataloader epoch as there we in the CutSet
        assert len(sampled_cuts) == len(eager_cuts)
        # Invariant 2: the items are not duplicated
        assert len(set(c.id for c in sampled_cuts)) == len(sampled_cuts)


@pytest.mark.parametrize(
    "sampler_cls", [SimpleCutSampler, BucketingSampler, DynamicCutSampler]
)
def test_sampler_filter(sampler_cls):
    # The dummy cuts have a duration of 1 second each
    cut_set = DummyManifest(CutSet, begin_id=0, end_id=100)
    sampler = sampler_cls(
        cut_set,
        shuffle=True,
        # Set an effective batch size of 10 cuts, as all have 1s duration == 100 frames
        # This way we're testing that it works okay when returning multiple batches in
        # a full epoch.
        max_duration=10.0,
    )
    removed_cut_id = "dummy-mono-cut-0010"
    sampler.filter(lambda cut: cut.id != removed_cut_id)
    sampled_cuts = []
    for batch in sampler:
        sampled_cuts.extend(batch)

    # The filtered cut is not there
    assert removed_cut_id in set(cut_set.ids)
    assert removed_cut_id not in set(c.id for c in sampled_cuts)

    # Invariant 1: we receive the same amount of items in a dataloader epoch as there we in the CutSet
    assert len(sampled_cuts) == len(cut_set) - 1
    # Invariant 2: the items are not duplicated
    assert len(set(c.id for c in sampled_cuts)) == len(sampled_cuts)


@pytest.mark.parametrize(
    "sampler_cls", [SimpleCutSampler, BucketingSampler, DynamicCutSampler]
)
def test_sampler_filter_called_twice_applies_both(sampler_cls):
    # The dummy cuts have a duration of 1 second each
    cut_set = DummyManifest(CutSet, begin_id=0, end_id=100)
    sampler = sampler_cls(
        cut_set,
        shuffle=True,
        # Set an effective batch size of 10 cuts, as all have 1s duration == 100 frames
        # This way we're testing that it works okay when returning multiple batches in
        # a full epoch.
        max_duration=10.0,
    )
    removed_cut_id1 = "dummy-mono-cut-0010"
    sampler.filter(lambda cut: cut.id != removed_cut_id1)
    removed_cut_id2 = "dummy-mono-cut-0011"
    sampler.filter(lambda cut: cut.id != removed_cut_id2)

    sampled_cuts = []
    for batch in sampler:
        sampled_cuts.extend(batch)

    # The filtered cut1 is not there
    assert removed_cut_id1 in set(cut_set.ids)
    assert removed_cut_id1 not in set(c.id for c in sampled_cuts)

    # The filtered cut2 is not there
    assert removed_cut_id2 in set(cut_set.ids)
    assert removed_cut_id2 not in set(c.id for c in sampled_cuts)

    # Invariant 1: we receive the same amount of items in a dataloader epoch as there we in the CutSet
    assert len(sampled_cuts) == len(cut_set) - 2
    # Invariant 2: the items are not duplicated
    assert len(set(c.id for c in sampled_cuts)) == len(sampled_cuts)


def test_cut_pairs_sampler_filter():
    # The dummy cuts have a duration of 1 second each
    cut_set = DummyManifest(CutSet, begin_id=0, end_id=100)
    sampler = CutPairsSampler(
        cut_set,
        cut_set,
        shuffle=True,
        # Set an effective batch size of 10 cuts, as all have 1s duration == 100 frames
        # This way we're testing that it works okay when returning multiple batches in
        # a full epoch.
        max_source_duration=100.0,
    )
    removed_cut_id = "dummy-mono-cut-0010"
    sampler.filter(lambda cut: cut.id != removed_cut_id)

    source_cuts, target_cuts = [], []
    for src_batch, tgt_batch in sampler:
        source_cuts.extend(src_batch)
        target_cuts.extend(tgt_batch)

    # The filtered cut is not there
    assert removed_cut_id in set(cut_set.ids)
    assert removed_cut_id not in set(c.id for c in source_cuts)

    # Invariant 1: we receive the same amount of items in a dataloader epoch as there we in the CutSet,
    # minus the filtered item
    assert len(source_cuts) == len(cut_set) - 1
    assert len(target_cuts) == len(cut_set) - 1
    # Invariant 2: the items are not duplicated
    assert len(set(c.id for c in source_cuts)) == len(source_cuts)
    assert len(set(c.id for c in target_cuts)) == len(target_cuts)


def test_zip_sampler_merge_batches_true():
    cuts1 = DummyManifest(CutSet, begin_id=0, end_id=100)
    cuts2 = DummyManifest(CutSet, begin_id=1000, end_id=1100)
    sampler = ZipSampler(
        # Note: each cut is 1s duration in this test.
        SimpleCutSampler(cuts1, max_duration=10),
        SimpleCutSampler(cuts2, max_duration=2),
    )
    batches = [b for b in sampler]
    assert len(batches) == 10
    for idx, batch in enumerate(batches):
        assert len(batch) == 12  # twelve 1s items
        assert (
            len([c for c in batch if 0 <= int(c.id.split("-")[-1]) <= 100]) == 10
        )  # ten come from cuts1
        assert (
            len([c for c in batch if 1000 <= int(c.id.split("-")[-1]) <= 1100]) == 2
        )  # two come from cuts2


def test_zip_sampler_cut_pairs_merge_batches_true():
    cuts1 = DummyManifest(CutSet, begin_id=0, end_id=100)
    cuts2 = DummyManifest(CutSet, begin_id=1000, end_id=1100)
    sampler = ZipSampler(
        # Note: each cut is 1s duration in this test.
        CutPairsSampler(cuts1, cuts1, max_source_duration=10),
        CutPairsSampler(cuts2, cuts2, max_source_duration=2),
    )
    batches = [b for b in sampler]
    assert len(batches) == 10
    for idx, (batch_src, batch_tgt) in enumerate(batches):
        assert len(batch_src) == len(batch_tgt)
        assert len(batch_src) == 12  # twelve 1s items
        assert (
            len([c for c in batch_src if 0 <= int(c.id.split("-")[-1]) <= 100]) == 10
        )  # ten come from cuts1
        assert (
            len([c for c in batch_src if 1000 <= int(c.id.split("-")[-1]) <= 1100]) == 2
        )  # two come from cuts2


def test_zip_sampler_merge_batches_false():
    cuts1 = DummyManifest(CutSet, begin_id=0, end_id=100)
    cuts2 = DummyManifest(CutSet, begin_id=1000, end_id=1100)
    sampler = ZipSampler(
        # Note: each cut is 1s duration in this test.
        SimpleCutSampler(cuts1, max_duration=10),
        SimpleCutSampler(cuts2, max_duration=2),
        merge_batches=False,
    )
    batches = [b for b in sampler]
    assert len(batches) == 10
    for idx, (batch_sampler1, batch_sampler2) in enumerate(batches):
        assert len(batch_sampler1) == 10
        assert (
            len([c for c in batch_sampler1 if 0 <= int(c.id.split("-")[-1]) <= 100])
            == 10
        )  # ten come from cuts1
        assert len(batch_sampler2) == 2
        assert (
            len([c for c in batch_sampler2 if 1000 <= int(c.id.split("-")[-1]) <= 1100])
            == 2
        )  # two come from cuts2


@pytest.mark.parametrize("randomize", [True, False])
def test_round_robin_sampler(randomize):
    cuts1 = DummyManifest(CutSet, begin_id=0, end_id=30)
    cuts2 = DummyManifest(CutSet, begin_id=1000, end_id=1100)
    sampler = RoundRobinSampler(
        # Note: each cut is 1s duration in this test.
        SimpleCutSampler(cuts1, max_duration=10),
        SimpleCutSampler(cuts2, max_duration=2),
        randomize=randomize,
    )

    batches = [b for b in sampler]
    assert len(batches) == 3 + 50

    batches_10cuts = [b for b in batches if len(b) == 10]
    assert len(batches_10cuts) == 3

    batches_2cuts = [b for b in batches if len(b) == 2]
    assert len(batches_2cuts) == 50

    if not randomize:
        assert len(batches[0]) == 10
        assert len(batches[1]) == 2
        assert len(batches[2]) == 10
        assert len(batches[3]) == 2
        assert len(batches[4]) == 10
        assert len(batches[5]) == 2
        assert len(batches[6]) == 2
        assert len(batches[7]) == 2
        assert len(batches[8]) == 2
        assert len(batches[9]) == 2
    # ... and so on


@pytest.mark.parametrize("sampler_cls", [SimpleCutSampler, DynamicCutSampler])
def test_single_cut_sampler_drop_last(sampler_cls):
    # The dummy cuts have a duration of 1 second each
    cut_set = DummyManifest(CutSet, begin_id=0, end_id=100)

    sampler = sampler_cls(
        cut_set,
        # Set an effective batch size of 15 cuts, as all have 1s duration == 100 frames
        # This way we're testing that it works okay when returning multiple batches in
        # a full epoch.
        max_duration=15.0,
        drop_last=True,
    )
    batches = []
    for batch in sampler:
        assert len(batch) == 15
        batches.append(batch)

    assert len(batches) == 6


SAMPLERS_FACTORIES_FOR_REPORT_TEST = [
    lambda: SimpleCutSampler(
        DummyManifest(CutSet, begin_id=0, end_id=10), max_cuts=10000
    ),
    lambda: DynamicCutSampler(
        DummyManifest(CutSet, begin_id=0, end_id=10), max_cuts=10000
    ),
    lambda: CutPairsSampler(
        DummyManifest(CutSet, begin_id=0, end_id=10),
        DummyManifest(CutSet, begin_id=0, end_id=10),
    ),
    lambda: BucketingSampler(
        DummyManifest(CutSet, begin_id=0, end_id=10), num_buckets=2, max_cuts=10000
    ),
    lambda: DynamicBucketingSampler(
        DummyManifest(CutSet, begin_id=0, end_id=10),
        max_duration=1.0,
        num_buckets=2,
    ),
    lambda: ZipSampler(
        SimpleCutSampler(DummyManifest(CutSet, begin_id=0, end_id=10), max_cuts=10000),
        SimpleCutSampler(DummyManifest(CutSet, begin_id=10, end_id=20), max_cuts=10000),
    ),
    lambda: RoundRobinSampler(
        SimpleCutSampler(DummyManifest(CutSet, begin_id=0, end_id=10), max_cuts=10000),
        SimpleCutSampler(DummyManifest(CutSet, begin_id=10, end_id=20), max_cuts=10000),
    ),
]


@pytest.mark.parametrize("create_sampler", SAMPLERS_FACTORIES_FOR_REPORT_TEST)
def test_sampler_get_report(create_sampler):
    sampler = create_sampler()
    sampler.filter(
        lambda c: "8" not in c.id
    )  # to check that the report correctly accounts for discarded data
    _ = [b for b in sampler]
    report = sampler.get_report()
    assert not report.startswith("Sampling statistics unavailable")
    assert "cuts discarded 0" not in report
    print(report)


@pytest.mark.parametrize("create_sampler", SAMPLERS_FACTORIES_FOR_REPORT_TEST)
def test_sampler_diagnostics_accumulate_across_epochs(create_sampler):
    sampler = create_sampler()
    sampler.filter(
        lambda c: "8" not in c.id
    )  # to check that the report correctly accounts for discarded data

    sampler.set_epoch(0)
    _ = [b for b in sampler]  # iterate full epoch
    diagnostics_ep0: SamplingDiagnostics = deepcopy(sampler.diagnostics)

    sampler.set_epoch(1)
    _ = [b for b in sampler]  # iterate full epoch
    diagnostics_ep1: SamplingDiagnostics = sampler.diagnostics

    # batch statistics
    assert diagnostics_ep0.kept_batches < diagnostics_ep1.kept_batches
    assert diagnostics_ep0.total_batches < diagnostics_ep1.total_batches
    # note: no assumption about diagnostics.num_discarded_batches here

    # cut statistics
    assert 2 * diagnostics_ep0.total_cuts == diagnostics_ep1.total_cuts
    assert 2 * diagnostics_ep0.kept_cuts == diagnostics_ep1.kept_cuts
    assert 2 * diagnostics_ep0.discarded_cuts == diagnostics_ep1.discarded_cuts


@pytest.mark.parametrize(
    "sampler_cls",
    [
        SimpleCutSampler,
        DynamicCutSampler,
    ],
)
def test_single_cut_sampler_lazy_shuffle(sampler_cls):
    # The dummy cuts have a duration of 1 second each
    cut_set = DummyManifest(CutSet, begin_id=0, end_id=100)
    with NamedTemporaryFile(suffix=".jsonl") as f:
        cut_set.to_jsonl(f.name)
        lazy_cuts = CutSet.from_jsonl_lazy(f.name)

        sampler = sampler_cls(
            lazy_cuts,
            shuffle=True,
            # Set an effective batch size of 10 cuts, as all have 1s duration == 100 frames
            # This way we're testing that it works okay when returning multiple batches in
            # a full epoch.
            max_duration=10.0,
        )
        sampled_cuts = []
        for batch in sampler:
            sampled_cuts.extend(batch)

        # Invariant 1: we receive the same amount of items in a dataloader epoch as there we in the CutSet
        assert len(sampled_cuts) == len(cut_set)
        # Invariant 2: the items are not duplicated
        assert len(set(c.id for c in sampled_cuts)) == len(sampled_cuts)
        # Invariant 3: the items are shuffled
        assert [c.id for c in sampled_cuts] != [c.id for c in lazy_cuts]


@pytest.mark.parametrize(
    "sampler_cls",
    [
        CutPairsSampler,
        pytest.param(
            BucketingSampler,
            marks=pytest.mark.xfail(
                reason="BucketingSampler does not support lazy cuts pairs."
            ),
        ),
    ],
)
def test_cut_pairs_sampler_lazy_shuffle(sampler_cls):
    # The dummy cuts have a duration of 1 second each
    cut_set = DummyManifest(CutSet, begin_id=0, end_id=100)
    with NamedTemporaryFile(suffix=".jsonl") as f:
        cut_set.to_jsonl(f.name)
        lazy_cuts = CutSet.from_jsonl_lazy(f.name)

        sampler = sampler_cls(
            lazy_cuts,
            lazy_cuts,
            shuffle=True,
            # Set an effective batch size of 10 cuts, as all have 1s duration == 100 frames
            # This way we're testing that it works okay when returning multiple batches in
            # a full epoch.
            max_source_duration=100.0,
        )
        sampled_src_cuts = []
        sampled_tgt_cuts = []
        for src_batch, tgt_batch in sampler:
            # Invariant 0: The order of source and target cut IDs is preserved within each batch.
            assert list(src_batch.ids) == list(tgt_batch.ids)
            sampled_src_cuts.extend(src_batch)
            sampled_tgt_cuts.extend(tgt_batch)

        # Invariant 1: we receive the same amount of items in a dataloader epoch as there we in the CutSet
        assert len(sampled_src_cuts) == len(cut_set)
        assert len(sampled_tgt_cuts) == len(cut_set)
        # Invariant 2: the items are not duplicated
        assert len(set(c.id for c in sampled_src_cuts)) == len(sampled_src_cuts)
        # Invariant 3: the items are shuffled
        assert [c.id for c in sampled_src_cuts] != [c.id for c in lazy_cuts]


def test_weighted_sampler_num_samples():
    cut_set = DummyManifest(CutSet, begin_id=0, end_id=100)
    weight = [random.random() for i in range(100)]
    num_samples = 32

    sampler = WeightedSimpleCutSampler(
        cut_set,
        weight,
        num_samples=num_samples,
        max_duration=10.0,
        drop_last=True,
    )

    sampled_cuts = []
    num_cuts = 0
    for batch in sampler:
        sampled_cuts.extend(batch)
        num_cuts += len(batch)

    assert num_cuts <= num_samples


def test_weighted_sampler_across_epochs():
    cut_set = DummyManifest(CutSet, begin_id=0, end_id=100)
    weight = [random.random() for i in range(100)]
    num_samples = 32

    sampler = WeightedSimpleCutSampler(
        cut_set,
        weight,
        num_samples=num_samples,
        max_duration=10.0,
        drop_last=True,
    )

    # 1st epoch
    sampler.set_epoch(1)
    batch = next(iter(sampler))
    cut_ids1 = [c.id for c in batch]

    # 2st epoch
    sampler.set_epoch(2)
    batch = next(iter(sampler))
    cut_ids2 = [c.id for c in batch]

    assert set(cut_ids1) != set(cut_ids2)


@pytest.mark.parametrize("datasize", [10, 1000, 20000])
@pytest.mark.parametrize("bufsize", [100, 1000, 10000])
def test_streaming_shuffle(datasize, bufsize):
    data = list(range(int(datasize)))
    shuffled = list(
        streaming_shuffle(iter(data), bufsize=int(bufsize), rng=random.Random(42))
    )
    assert len(data) == len(shuffled)
    assert len(shuffled) == len(set(shuffled))
    assert data != shuffled


@pytest.mark.parametrize(
    "sampler",
    [
        SimpleCutSampler(DummyManifest(CutSet, begin_id=0, end_id=10), max_cuts=1),
        CutPairsSampler(
            DummyManifest(CutSet, begin_id=0, end_id=10),
            DummyManifest(CutSet, begin_id=0, end_id=10),
            max_cuts=1,
        ),
        BucketingSampler(DummyManifest(CutSet, begin_id=0, end_id=10), max_cuts=1),
        ZipSampler(
            SimpleCutSampler(DummyManifest(CutSet, begin_id=0, end_id=10), max_cuts=1),
            SimpleCutSampler(DummyManifest(CutSet, begin_id=10, end_id=20), max_cuts=1),
        ),
        RoundRobinSampler(
            SimpleCutSampler(DummyManifest(CutSet, begin_id=0, end_id=5), max_cuts=1),
            SimpleCutSampler(DummyManifest(CutSet, begin_id=10, end_id=15), max_cuts=1),
        ),
    ],
)
def test_sampler_properties(sampler):
    assert sampler.remaining_cuts == 10
    assert isclose(sampler.remaining_duration, 10.0)
    assert sampler.num_cuts == 10
    batches = [b for b in sampler]
    assert sampler.remaining_cuts == 0
    assert isclose(sampler.remaining_duration, 0.0)
    assert sampler.num_cuts == 10


def test_report_padding_ratio_estimate():
    s = SimpleCutSampler(DummyManifest(CutSet, begin_id=0, end_id=1000), max_cuts=1)
    report_padding_ratio_estimate(s)  # just test that it runs


def test_time_constraint_strictness():
    strict = TimeConstraint(max_duration=100)

    # create cuts with large variance of durations
    cut_durs = [30.0, 30.0, 10.0, 10.0, 20.0]
    assert sum(cut_durs) == pytest.approx(100.0)
    cuts = [dummy_cut(idx, duration=cd) for idx, cd in enumerate(cut_durs)]

    strict.add(cuts[0])  # total duration: 30s, effective duration: 30s
    assert not strict.close_to_exceeding()
    assert not strict.exceeded()

    strict.add(cuts[1])  # total duration: 60s, effective duration: 60s
    assert not strict.close_to_exceeding()
    assert not strict.exceeded()

    strict.add(cuts[2])  # total duration: 70s, effective duration: 90s
    assert strict.close_to_exceeding()  # because 70s + longest seen 30s = 100s
    assert not strict.exceeded()

    strict.add(cuts[3])  # total duration: 80s, effective duration: 120s
    assert strict.close_to_exceeding()  # because 80s + longest seen 30s = 110s
    assert strict.exceeded()  # because longest seen 30s * 4 seen cuts = 120s


@pytest.mark.parametrize(
    "sampler_fn",
    [
        SimpleCutSampler,
        DynamicCutSampler,
        pytest.param(
            partial(BucketingSampler, num_buckets=2),
            marks=pytest.mark.xfail(
                reason="BucketingSampler will oversample cuts when world_size>1 and drop_last=False "
                "more than other samplers due to its implementation."
            ),
        ),
        partial(DynamicBucketingSampler, num_buckets=2),
    ],
)
@pytest.mark.parametrize(
    "world_size", [1, 2, 16, 32]
)  # 32 is more than 2x of available utterances
@pytest.mark.parametrize("batch_duration", [1, 2, 4, 8, 16])
def test_sampler_does_not_drop_cuts_with_multiple_ranks(
    sampler_fn, world_size, batch_duration
):
    cuts = DummyManifest(CutSet, begin_id=0, end_id=10)
    num_input_cuts = len(cuts)

    tot_cuts = []
    batches = []
    for rank in range(world_size):
        sampler = sampler_fn(
            cuts, max_duration=batch_duration, world_size=world_size, rank=rank
        )
        for batch in sampler:
            batches.append(batch)
            tot_cuts.extend(batch)

    def is_duplicate(cut):
        return re.search(r"^.+_dup\d+$", cut.id) is not None

    uniq_ids = [c.id for c in tot_cuts if not is_duplicate(c)]

    if world_size < num_input_cuts:
        # ws=1
        #   bs=1 => 10 (10batches)
        #   bs=2 => 10 (5batches)
        #   bs=4 => 10 (3batches)
        #   bs=8 => 10 (2batches)
        #   bs=16 => 10 (1batch)
        # ws=2
        #   bs=1 => 10 (1+1, 1+1, 1+1, 1+1, 1+1)
        #   bs=2 => 10 (2+2, 2+2, 1+1)
        #   bs=4 => 10 (4+4, 1+1)
        #   bs=8 => 10 (5+5)
        #   bs=16 => 10 (5+5)
        assert len(tot_cuts) == num_input_cuts
        assert len(uniq_ids) == len(tot_cuts)  # no duplicates
    else:
        # ws=16
        #   bs=1 => 16 (1x16, 6 duplicated)
        #   bs=2 => 16 (1x16, 6 duplicated)
        #   bs=4 => 16 (1x16, 6 duplicated)
        #   bs=8 => 16 (1x16, 6 duplicated)
        #   bs=16 => 16 (1x16, 6 duplicated)
        assert num_input_cuts < len(tot_cuts)
        assert len(tot_cuts) == world_size
        assert len(uniq_ids) == num_input_cuts
        assert len(tot_cuts) - len(uniq_ids) == world_size - num_input_cuts
        assert len(batches) == world_size
        assert all(len(b) == 1 for b in batches)


def test_sampler_map():
    cuts = DummyManifest(CutSet, begin_id=0, end_id=10)
    transform = CutConcatenate(gap=0.0, duration_factor=5.0)  # will glue 5 cuts into 1

    sampler = DynamicCutSampler(cuts, max_duration=5.0)
    sampler.map(transform)

    batches = [b for b in sampler]
    assert len(batches) == 2

    b = batches[0]
    assert len(b) == 1
    assert b[0].duration == 5.0

    b = batches[1]
    assert len(b) == 1
    assert b[0].duration == 5.0


def test_sampler_much_less_data_than_ddp_ranks():
    world_size = 128
    orig_cut = dummy_cut(0)
    cuts = CutSet([orig_cut])

    samplers = [
        DynamicCutSampler(
            cuts, max_cuts=256, drop_last=False, world_size=world_size, rank=i
        )
        for i in range(world_size)
    ]
    # None of the ranks drops anything, all of them return the one cut we have.
    for sampler in samplers:
        (batch,) = [b for b in sampler]
        assert len(batch) == 1
        (sampled_cut,) = batch
        assert (
            sampled_cut.id[: len(orig_cut.id)] == orig_cut.id
        )  # same stem, possibly added '_dupX' suffix
        # otherwise the cuts are identical
        sampled_cut.id = orig_cut.id
        assert sampled_cut == orig_cut
