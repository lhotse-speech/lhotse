import random
from itertools import islice

import pytest

from lhotse import CutSet
from lhotse.dataset.sampling.dynamic_bucketing import (
    DynamicBucketer,
    DynamicBucketingSampler,
    FixedBucketBatchSizeConstraint,
    estimate_duration_buckets,
)
from lhotse.testing.dummies import DummyManifest, dummy_cut


def test_estimate_duration_buckets_2b():
    cuts = DummyManifest(CutSet, begin_id=0, end_id=10)
    for i, c in enumerate(cuts):
        if i < 5:
            c.duration = 1
        else:
            c.duration = 2

    bins = estimate_duration_buckets(cuts, num_buckets=2)

    assert bins == [2]


def test_estimate_duration_buckets_4b():
    cuts = DummyManifest(CutSet, begin_id=0, end_id=20)
    for i, c in enumerate(cuts):
        if i < 5:
            c.duration = 1
        elif i < 10:
            c.duration = 2
        elif i < 15:
            c.duration = 3
        elif i < 20:
            c.duration = 4

    bins = estimate_duration_buckets(cuts, num_buckets=4)

    assert bins == [2, 3, 4]


def test_dynamic_bucketing_drop_last_false():
    cuts = DummyManifest(CutSet, begin_id=0, end_id=10)
    for i, c in enumerate(cuts):
        if i < 5:
            c.duration = 1
        else:
            c.duration = 2
    rng = random.Random(0)

    sampler = DynamicBucketer(
        cuts, duration_bins=[1.5], max_duration=5, rng=rng, world_size=1
    )
    batches = [b for b in sampler]
    sampled_cuts = [c for b in batches for c in b]

    # Invariant: no duplicated cut IDs
    assert len(set(c.id for b in batches for c in b)) == len(sampled_cuts)

    # Same number of sampled and source cuts.
    assert len(sampled_cuts) == len(cuts)

    # We sampled 4 batches with this RNG, like the following:
    assert len(batches) == 4

    assert len(batches[0]) == 2
    assert sum(c.duration for c in batches[0]) == 4

    assert len(batches[1]) == 2
    assert sum(c.duration for c in batches[1]) == 4

    assert len(batches[2]) == 5
    assert sum(c.duration for c in batches[2]) == 5

    assert len(batches[3]) == 1
    assert sum(c.duration for c in batches[3]) == 2


def test_dynamic_bucketing_drop_last_true():
    cuts = DummyManifest(CutSet, begin_id=0, end_id=10)
    for i, c in enumerate(cuts):
        if i < 5:
            c.duration = 1
        else:
            c.duration = 2
    rng = random.Random(0)

    sampler = DynamicBucketer(
        cuts, duration_bins=[1.5], max_duration=5, rng=rng, drop_last=True, world_size=1
    )
    batches = [b for b in sampler]
    sampled_cuts = [c for b in batches for c in b]

    # Invariant: no duplicated cut IDs.
    assert len(set(c.id for b in batches for c in b)) == len(sampled_cuts)

    # Some cuts were not sampled due to drop_last.
    assert len(sampled_cuts) < len(cuts)
    assert len(sampled_cuts) == 9

    # We sampled 3 batches with this RNG, like the following:
    assert len(batches) == 3

    assert len(batches[0]) == 2
    assert sum(c.duration for c in batches[0]) == 4

    assert len(batches[1]) == 2
    assert sum(c.duration for c in batches[1]) == 4

    assert len(batches[2]) == 5
    assert sum(c.duration for c in batches[2]) == 5


@pytest.mark.parametrize("concurrent", [False, True])
def test_dynamic_bucketing_sampler(concurrent):
    cuts = DummyManifest(CutSet, begin_id=0, end_id=10)
    for i, c in enumerate(cuts):
        if i < 5:
            c.duration = 1
        else:
            c.duration = 2

    sampler = DynamicBucketingSampler(
        cuts, max_duration=5, duration_bins=[1.5], seed=0, concurrent=concurrent
    )
    batches = [b for b in sampler]
    sampled_cuts = [c for b in batches for c in b]

    # Invariant: no duplicated cut IDs
    assert len(set(c.id for b in batches for c in b)) == len(sampled_cuts)

    # Same number of sampled and source cuts.
    assert len(sampled_cuts) == len(cuts)

    # We sampled 4 batches with this RNG, like the following:
    assert len(batches) == 4

    assert len(batches[0]) == 2
    assert sum(c.duration for c in batches[0]) == 4

    assert len(batches[1]) == 5
    assert sum(c.duration for c in batches[1]) == 5

    assert len(batches[2]) == 2
    assert sum(c.duration for c in batches[2]) == 4

    assert len(batches[3]) == 1
    assert sum(c.duration for c in batches[3]) == 2


def test_dynamic_bucketing_sampler_precomputed_duration_bins():
    cuts = DummyManifest(CutSet, begin_id=0, end_id=10)
    for i, c in enumerate(cuts):
        if i < 5:
            c.duration = 1
        else:
            c.duration = 2

    # all cuts actually go into bucket 1 and bucket 0 is always empty
    sampler = DynamicBucketingSampler(
        cuts,
        max_duration=5,
        num_buckets=2,
        duration_bins=[0.5],
        seed=0,
        shuffle=True,
    )
    next(iter(sampler))

    batches = [b for b in sampler]
    sampled_cuts = [c for b in batches for c in b]

    # Invariant: no duplicated cut IDs
    assert len(set(c.id for b in batches for c in b)) == len(sampled_cuts)

    # Same number of sampled and source cuts.
    assert len(sampled_cuts) == len(cuts)

    # We sampled 5 batches with this RNG, like the following:
    assert len(batches) == 4

    assert len(batches[0]) == 2
    assert sum(c.duration for c in batches[0]) == 4

    assert len(batches[1]) == 4
    assert sum(c.duration for c in batches[1]) == 5

    assert len(batches[2]) == 2
    assert sum(c.duration for c in batches[2]) == 3

    assert len(batches[3]) == 2
    assert sum(c.duration for c in batches[3]) == 3


def test_dynamic_bucketing_sampler_max_duration_and_max_cuts():
    cuts = DummyManifest(CutSet, begin_id=0, end_id=10)
    for i, c in enumerate(cuts):
        if i < 5:
            c.duration = 1
        else:
            c.duration = 2

    sampler = DynamicBucketingSampler(
        cuts, max_duration=5, max_cuts=1, num_buckets=2, seed=0
    )
    batches = [b for b in sampler]
    sampled_cuts = [c for b in batches for c in b]

    # Invariant: no duplicated cut IDs
    assert len(set(c.id for b in batches for c in b)) == len(sampled_cuts)

    # Same number of sampled and source cuts.
    assert len(sampled_cuts) == len(cuts)

    # We sampled 10 batches because max_cuts == 1
    assert len(batches) == 10
    for b in batches:
        assert len(b) == 1


def test_dynamic_bucketing_sampler_too_small_data_can_be_sampled():
    cuts = DummyManifest(CutSet, begin_id=0, end_id=10)
    for i, c in enumerate(cuts):
        if i < 5:
            c.duration = 1
        else:
            c.duration = 2

    # 10 cuts with 30s total are not enough to satisfy max_duration of 100 with 2 buckets
    sampler = DynamicBucketingSampler(
        cuts, max_duration=100, duration_bins=[1.5], seed=0
    )
    batches = [b for b in sampler]
    sampled_cuts = [c for b in batches for c in b]

    # Invariant: no duplicated cut IDs
    assert len(set(c.id for b in batches for c in b)) == len(sampled_cuts)

    # Same number of sampled and source cuts.
    assert len(sampled_cuts) == len(cuts)

    # We sampled 10 batches
    assert len(batches) == 2

    # Each batch has five cuts
    for b in batches:
        assert len(b) == 5


def test_dynamic_bucketing_sampler_much_less_data_than_ddp_ranks():
    world_size = 128
    orig_cut = dummy_cut(0)
    cuts = CutSet([orig_cut])
    samplers = [
        DynamicBucketingSampler(
            cuts,
            max_duration=2000.0,
            duration_bins=[1.5, 3.7, 15.2, 27.9, 40.0],
            drop_last=False,
            concurrent=False,
            world_size=world_size,
            rank=i,
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


def test_dynamic_bucketing_sampler_too_small_data_drop_last_true_results_in_no_batches():
    cuts = DummyManifest(CutSet, begin_id=0, end_id=10)
    for i, c in enumerate(cuts):
        if i < 5:
            c.duration = 1
        else:
            c.duration = 2

    # 10 cuts with 30s total are not enough to satisfy max_duration of 100 with 2 buckets
    sampler = DynamicBucketingSampler(
        cuts, max_duration=100, num_buckets=2, seed=0, drop_last=True
    )
    batches = [b for b in sampler]
    assert len(batches) == 0


def test_dynamic_bucketing_sampler_filter():
    cuts = DummyManifest(CutSet, begin_id=0, end_id=10)
    for i, c in enumerate(cuts):
        if i < 5:
            c.duration = 1
        else:
            c.duration = 2

    sampler = DynamicBucketingSampler(cuts, max_duration=5, num_buckets=2, seed=0)
    sampler.filter(lambda cut: cut.duration > 1)
    batches = [b for b in sampler]
    sampled_cuts = [c for b in batches for c in b]

    # Invariant: no duplicated cut IDs
    assert len(set(c.id for b in batches for c in b)) == len(sampled_cuts)

    # Same number of sampled and source cuts.
    assert len(sampled_cuts) < len(cuts)
    assert len(sampled_cuts) == 5

    # We sampled 4 batches with this RNG, like the following:
    assert len(batches) == 3

    assert len(batches[0]) == 2
    assert sum(c.duration for c in batches[0]) == 4

    assert len(batches[1]) == 2
    assert sum(c.duration for c in batches[1]) == 4

    assert len(batches[2]) == 1
    assert sum(c.duration for c in batches[2]) == 2


def test_dynamic_bucketing_sampler_shuffle():
    cuts = DummyManifest(CutSet, begin_id=0, end_id=10)
    for i, c in enumerate(cuts):
        if i < 5:
            c.duration = 1
        else:
            c.duration = 2

    sampler = DynamicBucketingSampler(
        cuts, max_duration=5, num_buckets=2, seed=0, shuffle=True
    )

    epoch_batches = []
    for epoch in range(2):
        sampler.set_epoch(epoch)

        batches = [b for b in sampler]
        sampled_cuts = [c for b in batches for c in b]

        # Invariant: no duplicated cut IDs
        assert len(set(c.id for b in batches for c in b)) == len(sampled_cuts)

        # Invariant: Same number of sampled and source cuts.
        assert len(sampled_cuts) == len(cuts)

        epoch_batches.append(batches)

    # Epoch 0 batches are different than epoch 1 batches
    assert epoch_batches[0] != epoch_batches[1]


def test_dynamic_bucketing_sampler_cut_pairs():
    cuts = DummyManifest(CutSet, begin_id=0, end_id=10)
    for i, c in enumerate(cuts):
        if i < 5:
            c.duration = 1
        else:
            c.duration = 2

    sampler = DynamicBucketingSampler(
        cuts, cuts, max_duration=5, duration_bins=[1.5], seed=0
    )
    batches = [b for b in sampler]
    sampled_cut_pairs = [cut_pair for b in batches for cut_pair in zip(*b)]
    source_cuts = [sc for sc, tc in sampled_cut_pairs]
    target_cuts = [tc for sc, tc in sampled_cut_pairs]

    # Invariant: no duplicated cut IDs
    assert len(set(c.id for c in source_cuts)) == len(cuts)
    assert len(set(c.id for c in target_cuts)) == len(cuts)

    # Same number of sampled and source cuts.
    assert len(sampled_cut_pairs) == len(cuts)

    # We sampled 4 batches with this RNG, like the following:
    assert len(batches) == 4

    bidx = 0
    sc, tc = batches[bidx][0], batches[bidx][1]
    assert len(sc) == 2
    assert len(tc) == 2
    assert sum(c.duration for c in sc) == 4
    assert sum(c.duration for c in tc) == 4

    bidx = 1
    sc, tc = batches[bidx][0], batches[bidx][1]
    assert len(sc) == 5
    assert len(tc) == 5
    assert sum(c.duration for c in sc) == 5
    assert sum(c.duration for c in tc) == 5

    bidx = 2
    sc, tc = batches[bidx][0], batches[bidx][1]
    assert len(sc) == 2
    assert len(tc) == 2
    assert sum(c.duration for c in sc) == 4
    assert sum(c.duration for c in tc) == 4

    bidx = 3
    sc, tc = batches[bidx][0], batches[bidx][1]
    assert len(sc) == 1
    assert len(tc) == 1
    assert sum(c.duration for c in sc) == 2
    assert sum(c.duration for c in tc) == 2


def test_dynamic_bucketing_sampler_cut_pairs_shuffle():
    cuts = DummyManifest(CutSet, begin_id=0, end_id=10)
    for i, c in enumerate(cuts):
        if i < 5:
            c.duration = 1
        else:
            c.duration = 2

    sampler = DynamicBucketingSampler(
        cuts, cuts, max_duration=5, num_buckets=2, seed=0, shuffle=True
    )

    epoch_batches = []
    for epoch in range(2):
        sampler.set_epoch(epoch)

        batches = [b for b in sampler]
        sampled_cut_pairs = [cut_pair for b in batches for cut_pair in zip(*b)]
        source_cuts = [sc for sc, tc in sampled_cut_pairs]
        target_cuts = [tc for sc, tc in sampled_cut_pairs]

        # Invariant: no duplicated cut IDs
        assert len(set(c.id for c in source_cuts)) == len(cuts)
        assert len(set(c.id for c in target_cuts)) == len(cuts)

        # Same number of sampled and source cuts.
        assert len(sampled_cut_pairs) == len(cuts)

        epoch_batches.append(batches)

    # Epoch 0 batches are different than epoch 1 batches
    assert epoch_batches[0] != epoch_batches[1]


def test_dynamic_bucketing_sampler_cut_pairs_filter():
    cuts = DummyManifest(CutSet, begin_id=0, end_id=10)
    for i, c in enumerate(cuts):
        if i < 5:
            c.duration = 1
        else:
            c.duration = 2

    sampler = DynamicBucketingSampler(cuts, cuts, max_duration=5, num_buckets=2, seed=0)
    sampler.filter(lambda c: c.duration > 1)
    batches = [b for b in sampler]
    sampled_cut_pairs = [cut_pair for b in batches for cut_pair in zip(*b)]
    source_cuts = [sc for sc, tc in sampled_cut_pairs]
    target_cuts = [tc for sc, tc in sampled_cut_pairs]

    # Invariant: no duplicated cut IDs (there are 5 unique IDs)
    assert len(set(c.id for c in source_cuts)) == 5
    assert len(set(c.id for c in target_cuts)) == 5

    # Smaller number of sampled cuts than the source cuts.
    assert len(sampled_cut_pairs) < len(cuts)
    assert len(sampled_cut_pairs) == 5

    # We sampled 3 batches with this RNG, like the following:
    assert len(batches) == 3

    bidx = 0
    sc, tc = batches[bidx][0], batches[bidx][1]
    assert len(sc) == 2
    assert len(tc) == 2
    assert sum(c.duration for c in sc) == 4
    assert sum(c.duration for c in tc) == 4

    bidx = 1
    sc, tc = batches[bidx][0], batches[bidx][1]
    assert len(sc) == 2
    assert len(tc) == 2
    assert sum(c.duration for c in sc) == 4
    assert sum(c.duration for c in tc) == 4

    bidx = 2
    sc, tc = batches[bidx][0], batches[bidx][1]
    assert len(sc) == 1
    assert len(tc) == 1
    assert sum(c.duration for c in sc) == 2
    assert sum(c.duration for c in tc) == 2


def test_dynamic_bucketing_sampler_cut_triplets():
    cuts = DummyManifest(CutSet, begin_id=0, end_id=10)
    for i, c in enumerate(cuts):
        if i < 5:
            c.duration = 1
        else:
            c.duration = 2

    sampler = DynamicBucketingSampler(
        cuts, cuts, cuts, max_duration=5, duration_bins=[1.5], seed=0
    )
    batches = [b for b in sampler]
    sampled_cut_triplets = [cut_triplet for b in batches for cut_triplet in zip(*b)]
    cuts1 = [c1 for c1, c2, c3 in sampled_cut_triplets]
    cuts2 = [c2 for c1, c2, c3 in sampled_cut_triplets]
    cuts3 = [c3 for c1, c2, c3 in sampled_cut_triplets]

    # Invariant: no duplicated cut IDs
    assert len(set(c.id for c in cuts1)) == len(cuts)
    assert len(set(c.id for c in cuts2)) == len(cuts)
    assert len(set(c.id for c in cuts3)) == len(cuts)

    # Same number of sampled and source cuts.
    assert len(sampled_cut_triplets) == len(cuts)

    # We sampled 4 batches with this RNG, like the following:
    assert len(batches) == 4

    bidx = 0
    c1, c2, c3 = batches[bidx][0], batches[bidx][1], batches[bidx][2]
    assert len(c1) == 2
    assert len(c2) == 2
    assert len(c3) == 2
    assert sum(c.duration for c in c1) == 4
    assert sum(c.duration for c in c2) == 4
    assert sum(c.duration for c in c3) == 4

    bidx = 1
    c1, c2, c3 = batches[bidx][0], batches[bidx][1], batches[bidx][2]
    assert len(c1) == 5
    assert len(c2) == 5
    assert len(c3) == 5
    assert sum(c.duration for c in c1) == 5
    assert sum(c.duration for c in c2) == 5
    assert sum(c.duration for c in c3) == 5

    bidx = 2
    c1, c2, c3 = batches[bidx][0], batches[bidx][1], batches[bidx][2]
    assert len(c1) == 2
    assert len(c2) == 2
    assert len(c3) == 2
    assert sum(c.duration for c in c1) == 4
    assert sum(c.duration for c in c2) == 4
    assert sum(c.duration for c in c3) == 4

    bidx = 3
    c1, c2, c3 = batches[bidx][0], batches[bidx][1], batches[bidx][2]
    assert len(c1) == 1
    assert len(c2) == 1
    assert len(c3) == 1
    assert sum(c.duration for c in c1) == 2
    assert sum(c.duration for c in c2) == 2
    assert sum(c.duration for c in c3) == 2


def test_dynamic_bucketing_quadratic_duration():
    cuts = DummyManifest(CutSet, begin_id=0, end_id=10)
    for i, c in enumerate(cuts):
        if i < 5:
            c.duration = 5
        else:
            c.duration = 30

    # Set max_duration to 61 so that with quadratic_duration disabled,
    # we would have gotten 2 per batch, but otherwise we only get 1.

    # quadratic_duration=30
    sampler = DynamicBucketingSampler(
        cuts, max_duration=61, duration_bins=[10.0], seed=0, quadratic_duration=30
    )
    batches = [b for b in sampler]
    assert len(batches) == 6
    for b in batches[:5]:
        assert len(b) == 1  # single cut
        assert b[0].duration == 30  # 30s long

    b = batches[5]
    assert len(b) == 5  # 5 cuts
    assert sum(c.duration for c in b) == 25  # each 5s long

    # quadratic_duration=None (disabled)
    sampler = DynamicBucketingSampler(
        cuts, max_duration=61, duration_bins=[10.0], seed=0, quadratic_duration=None
    )
    batches = [b for b in sampler]
    assert len(batches) == 4
    for b in batches[:2]:
        assert len(b) == 2  # two cuts
        assert sum(c.duration for c in b) == 60  # each 30s long

    b = batches[2]
    assert len(b) == 5  # 5 cuts
    assert sum(c.duration for c in b) == 25  # each 5s long

    b = batches[3]
    assert len(b) == 1  # single cut
    assert sum(c.duration for c in b) == 30  # 30s long


@pytest.mark.parametrize("sync_buckets", [True, False])
def test_dynamic_bucketing_sampler_sync_buckets_iterable_dataset_usage(sync_buckets):
    # With iterable datasets a sampler replica will be placed in each dataloading worker,
    # given world_size=1, and have its data shuffled differently than other replicas.
    # To simulate that in this test, we provide a different seed and rank=0 world_size=1.
    dur_rng = random.Random(0)
    cuts = CutSet(
        [
            dummy_cut(i, duration=dur_rng.choices([1, 10], weights=[0.9, 0.1])[0])
            for i in range(10000)
        ]
    )

    common = dict(
        max_duration=5,
        num_buckets=2,
        rank=0,
        sync_buckets=sync_buckets,
        world_size=1,
        drop_last=True,
        shuffle=True,
        duration_bins=[5.0],
    )
    s0 = DynamicBucketingSampler(cuts, seed=0, **common)
    s1 = DynamicBucketingSampler(cuts, seed=1, **common)

    # check the first 30 mini-batches
    batches0 = [b for b in islice(s0, 30)]
    batches1 = [b for b in islice(s1, 30)]
    cuts0 = CutSet([c for b in batches0 for c in b])
    cuts1 = CutSet([c for b in batches1 for c in b])

    # Invariant: no duplicated cut IDs across ranks
    assert set(cuts0.ids) & set(cuts1.ids) == set()

    if sync_buckets:
        matching_ids = []
        # Ensure identical batch sizes and example durations
        for bidx, (b0, b1) in enumerate(zip(batches0, batches1)):
            assert len(b0) == len(b1), bidx
            for c0, c1 in zip(b0, b1):
                assert c0.duration == c1.duration
                matching_ids.append(c0.id == c1.id)
        # At least some IDs are mismatching because despite identical shapes, the actual sampled data is different.
        assert not all(matching_ids)
    if not sync_buckets:
        # some shapes will be mismatched because different buckets were selected.
        matching_shapes = [len(b0) == len(b1) for b0, b1 in zip(batches0, batches1)]
        assert not all(matching_shapes)


@pytest.mark.parametrize("sync_buckets", [True, False])
def test_dynamic_bucketing_sampler_sync_buckets_map_dataset_usage(sync_buckets):
    # With map datasets the sampler lives in the training loop process and must have synced random seed
    # with other ranks in DDP.
    # The data is de-duplicated by sampling world_size batches and keeping the batch at rank index.
    # To simulate that in this test, we provide the same seed, world_size=2 and set rank appropriately.
    dur_rng = random.Random(0)
    cuts = CutSet(
        [
            dummy_cut(i, duration=dur_rng.choices([1, 10], weights=[0.9, 0.1])[0])
            for i in range(10000)
        ]
    )

    common = dict(
        max_duration=5,
        num_buckets=2,
        seed=0,
        sync_buckets=sync_buckets,
        world_size=2,
        drop_last=True,
        shuffle=True,
        duration_bins=[5.0],
    )
    s0 = DynamicBucketingSampler(cuts, rank=0, **common)
    s1 = DynamicBucketingSampler(cuts, rank=1, **common)

    # check the first 30 mini-batches
    batches0 = [b for b in islice(s0, 30)]
    batches1 = [b for b in islice(s1, 30)]
    cuts0 = CutSet([c for b in batches0 for c in b])
    cuts1 = CutSet([c for b in batches1 for c in b])

    # Invariant: no duplicated cut IDs across ranks
    assert set(cuts0.ids) & set(cuts1.ids) == set()

    if sync_buckets:
        matching_ids = []
        # Ensure identical batch sizes and example durations
        for bidx, (b0, b1) in enumerate(zip(batches0, batches1)):
            assert len(b0) == len(b1), bidx
            for c0, c1 in zip(b0, b1):
                assert c0.duration == c1.duration
                matching_ids.append(c0.id == c1.id)
        # At least some IDs are mismatching because despite identical shapes, the actual sampled data is different.
        assert not all(matching_ids)
    if not sync_buckets:
        # some shapes will be mismatched because different buckets were selected.
        matching_shapes = [len(b0) == len(b1) for b0, b1 in zip(batches0, batches1)]
        assert not all(matching_shapes)


def test_dynamic_bucketing_sampler_fixed_batch_constraint():
    cuts = DummyManifest(CutSet, begin_id=0, end_id=10)
    for i, c in enumerate(cuts):
        if i < 5:
            c.duration = 1
        else:
            c.duration = 2

    duration_bins = [1.5, 2.5]
    sampler = DynamicBucketingSampler(
        cuts,
        duration_bins=duration_bins,
        constraint=FixedBucketBatchSizeConstraint(
            max_seq_len_buckets=duration_bins, batch_sizes=[2, 1]
        ),
        seed=0,
        shuffle=True,
    )

    batches = [b for b in sampler]
    sampled_cuts = [c for b in batches for c in b]

    # Invariant: no duplicated cut IDs
    assert len(set(c.id for b in batches for c in b)) == len(sampled_cuts)

    # Same number of sampled and source cuts.
    assert len(sampled_cuts) == len(cuts)

    # We sampled the follwoing batches with this RNG:
    assert len(batches) == 8
    print([len(b) for b in batches])

    assert len(batches[0]) == 1
    assert sum(c.duration for c in batches[0]) == 2

    assert len(batches[1]) == 2
    assert sum(c.duration for c in batches[1]) == 2

    assert len(batches[2]) == 2
    assert sum(c.duration for c in batches[2]) == 2

    assert len(batches[3]) == 1
    assert sum(c.duration for c in batches[3]) == 2

    assert len(batches[4]) == 1
    assert sum(c.duration for c in batches[4]) == 2

    assert len(batches[5]) == 1
    assert sum(c.duration for c in batches[5]) == 2

    assert len(batches[6]) == 1
    assert sum(c.duration for c in batches[6]) == 2

    assert len(batches[7]) == 1
    assert sum(c.duration for c in batches[7]) == 1


def test_select_bucket_includes_upper_bound_in_bin():
    constraint = FixedBucketBatchSizeConstraint(
        max_seq_len_buckets=[2.0, 4.0], batch_sizes=[2, 1]
    )

    # within bounds
    assert (
        constraint.select_bucket(constraint.max_seq_len_buckets, example_len=1.0) == 0
    )
    assert (
        constraint.select_bucket(constraint.max_seq_len_buckets, example_len=2.0) == 0
    )
    assert (
        constraint.select_bucket(constraint.max_seq_len_buckets, example_len=3.0) == 1
    )
    assert (
        constraint.select_bucket(constraint.max_seq_len_buckets, example_len=4.0) == 1
    )
    constraint.add(dummy_cut(0, duration=4.0))  # can add max duration without exception

    # out of bounds
    assert (
        constraint.select_bucket(constraint.max_seq_len_buckets, example_len=5.0) == 2
    )
    with pytest.raises(AssertionError):
        constraint.add(dummy_cut(0, duration=5.0))
