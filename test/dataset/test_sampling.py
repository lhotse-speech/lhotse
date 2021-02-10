import pytest

from lhotse import CutSet
from lhotse.dataset.sampling import BucketingSampler, CutPairsSampler, SingleCutSampler
from lhotse.dataset.transforms import concat_cuts
from lhotse.testing.dummies import DummyManifest, dummy_cut


@pytest.fixture
def libri_cut_set():
    cs = CutSet.from_json('test/fixtures/libri/cuts.json')
    return CutSet.from_cuts([
        cs[0],
        cs[0].with_id('copy-1'),
        cs[0].with_id('copy-2'),
        cs[0].append(cs[0])
    ])


def test_single_cut_sampler_shuffling():
    # The dummy cuts have a duration of 1 second each
    cut_set = DummyManifest(CutSet, begin_id=0, end_id=100)

    sampler = SingleCutSampler(
        cut_set,
        shuffle=True,
        # Set an effective batch size of 10 cuts, as all have 1s duration == 100 frames
        # This way we're testing that it works okay when returning multiple batches in
        # a full epoch.
        max_frames=1000
    )
    sampler_cut_ids = []
    for batch in sampler:
        sampler_cut_ids.extend(batch)

    # Invariant 1: we receive the same amount of items in a dataloader epoch as there we in the CutSet
    assert len(sampler_cut_ids) == len(cut_set)
    # Invariant 2: the items are not duplicated
    assert len(set(sampler_cut_ids)) == len(sampler_cut_ids)
    # Invariant 3: the items are shuffled, i.e. the order is different than that in the CutSet
    assert sampler_cut_ids != [c.id for c in cut_set]


def test_single_cut_sampler_low_max_frames(libri_cut_set):
    sampler = SingleCutSampler(libri_cut_set, shuffle=False, max_frames=2)
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
        max_source_frames=1000,
        max_target_frames=500,
    )
    sampler_cut_ids = []
    for batch in sampler:
        sampler_cut_ids.extend(batch)

    # Invariant 1: we receive the same amount of items in a dataloader epoch as there we in the CutSet
    assert len(sampler_cut_ids) == len(cut_set)
    # Invariant 2: the items are not duplicated
    assert len(set(sampler_cut_ids)) == len(sampler_cut_ids)
    # Invariant 3: the items are shuffled, i.e. the order is different than that in the CutSet
    assert sampler_cut_ids != [c.id for c in cut_set]


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


def test_bucketing_sampler_single_cuts():
    cut_set = DummyManifest(CutSet, begin_id=0, end_id=1000)
    sampler = BucketingSampler(cut_set, sampler_type=SingleCutSampler)
    cut_ids = []
    for batch in sampler:
        cut_ids.extend(batch)
    assert set(cut_set.ids) == set(cut_ids)


def test_bucketing_sampler_shuffle():
    cut_set = DummyManifest(CutSet, begin_id=0, end_id=10)
    sampler = BucketingSampler(cut_set, sampler_type=SingleCutSampler, shuffle=True, num_buckets=2, max_frames=200)

    sampler.set_epoch(0)
    batches_ep0 = []
    for batch in sampler:
        # Convert List[str] to Tuple[str, ...] so that it's hashable
        batches_ep0.append(tuple(batch))
    assert set(cut_set.ids) == set(cid for batch in batches_ep0 for cid in batch)

    sampler.set_epoch(1)
    batches_ep1 = []
    for batch in sampler:
        batches_ep1.append(tuple(batch))
    assert set(cut_set.ids) == set(cid for batch in batches_ep1 for cid in batch)

    # BucketingSampler ordering may be different in different epochs (=> use set() to make it irrelevant)
    # Internal sampler (SingleCutSampler) ordering should be different in different epochs
    assert set(batches_ep0) != set(batches_ep1)


def test_bucketing_sampler_cut_pairs():
    cut_set1 = DummyManifest(CutSet, begin_id=0, end_id=1000)
    cut_set2 = DummyManifest(CutSet, begin_id=0, end_id=1000)
    sampler = BucketingSampler(cut_set1, cut_set2, sampler_type=CutPairsSampler)

    cut_ids = []
    for batch in sampler:
        cut_ids.extend(batch)
    assert set(cut_set1.ids) == set(cut_ids)
    assert set(cut_set2.ids) == set(cut_ids)


def test_bucketing_sampler_buckets_have_different_durations():
    cut_set_1s = DummyManifest(CutSet, begin_id=0, end_id=10)
    cut_set_2s = DummyManifest(CutSet, begin_id=10, end_id=20)
    for c in cut_set_2s:
        c.duration = 2.0
    cut_set = cut_set_1s + cut_set_2s

    # The bucketing sampler should return 5 batches with two 1s cuts, and 10 batches with one 2s cut.
    sampler = BucketingSampler(
        cut_set,
        sampler_type=SingleCutSampler,
        max_frames=200,
        num_buckets=2
    )
    batches = list(sampler)
    assert len(batches) == 15

    # All cuts have the same durations (i.e. are from the same bucket in this case)
    for batch in batches:
        batch_durs = [cut_set[cid].duration for cid in batch]
        assert all(d == batch_durs[0] for d in batch_durs)

    batches = sorted(batches, key=len)
    assert all(len(b) == 1 for b in batches[:10])
    assert all(len(b) == 2 for b in batches[10:])
