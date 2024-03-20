import random

from lhotse import CutSet
from lhotse.dataset.sampling.dynamic_bucketing import (
    DynamicBucketer,
    DynamicBucketingSampler,
    estimate_duration_buckets,
)
from lhotse.testing.dummies import DummyManifest


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

    sampler = DynamicBucketer(cuts, duration_bins=[2], max_duration=5, rng=rng)
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
        cuts, duration_bins=[2], max_duration=5, rng=rng, drop_last=True
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


def test_dynamic_bucketing_sampler():
    cuts = DummyManifest(CutSet, begin_id=0, end_id=10)
    for i, c in enumerate(cuts):
        if i < 5:
            c.duration = 1
        else:
            c.duration = 2

    sampler = DynamicBucketingSampler(cuts, max_duration=5, num_buckets=2, seed=0)
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

    assert len(batches[1]) == 2
    assert sum(c.duration for c in batches[1]) == 3

    assert len(batches[2]) == 2
    assert sum(c.duration for c in batches[2]) == 3

    assert len(batches[3]) == 4
    assert sum(c.duration for c in batches[3]) == 5


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
    sampler = DynamicBucketingSampler(cuts, max_duration=100, num_buckets=2, seed=0)
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

    sampler = DynamicBucketingSampler(cuts, cuts, max_duration=5, num_buckets=2, seed=0)
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
    assert len(sc) == 2
    assert len(tc) == 2
    assert sum(c.duration for c in sc) == 4
    assert sum(c.duration for c in tc) == 4

    bidx = 2
    sc, tc = batches[bidx][0], batches[bidx][1]
    assert len(sc) == 5
    assert len(tc) == 5
    assert sum(c.duration for c in sc) == 5
    assert sum(c.duration for c in tc) == 5

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
        cuts, cuts, cuts, max_duration=5, num_buckets=2, seed=0
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
    assert len(c1) == 2
    assert len(c2) == 2
    assert len(c3) == 2
    assert sum(c.duration for c in c1) == 4
    assert sum(c.duration for c in c2) == 4
    assert sum(c.duration for c in c3) == 4

    bidx = 2
    c1, c2, c3 = batches[bidx][0], batches[bidx][1], batches[bidx][2]
    assert len(c1) == 5
    assert len(c2) == 5
    assert len(c3) == 5
    assert sum(c.duration for c in c1) == 5
    assert sum(c.duration for c in c2) == 5
    assert sum(c.duration for c in c3) == 5

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
        cuts, max_duration=61, num_buckets=2, seed=0, quadratic_duration=30
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
        cuts, max_duration=61, num_buckets=2, seed=0, quadratic_duration=None
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
