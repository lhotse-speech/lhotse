import random
from itertools import groupby
from tempfile import NamedTemporaryFile

import pytest

from lhotse import CutSet
from lhotse.dataset.cut_transforms import concat_cuts
from lhotse.dataset.sampling import BucketingSampler, CutPairsSampler, SingleCutSampler, ZipSampler, streaming_shuffle
from lhotse.testing.dummies import DummyManifest, dummy_cut
from lhotse.utils import nullcontext as does_not_raise


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
    sampled_cuts = []
    for batch in sampler:
        sampled_cuts.extend(batch)

    # Invariant 1: we receive the same amount of items in a dataloader epoch as there we in the CutSet
    assert len(sampled_cuts) == len(cut_set)
    # Invariant 2: the items are not duplicated
    assert len(set(c.id for c in sampled_cuts)) == len(sampled_cuts)
    # Invariant 3: the items are shuffled, i.e. the order is different than that in the CutSet
    assert [c.id for c in sampled_cuts] != [c.id for c in cut_set]


@pytest.mark.parametrize(
    ['max_duration', 'max_frames', 'max_samples', 'exception_expectation'],
    [
        (None, None, None, does_not_raise()),  # represents no criterion (unlimited batch size)
        (10.0, None, None, does_not_raise()),
        (None, 1000, None, does_not_raise()),
        (None, None, 160000, does_not_raise()),
        (None, 1000, 160000, pytest.raises(AssertionError)),
        (5.0, 1000, 160000, pytest.raises(AssertionError)),
    ]
)
def test_single_cut_sampler_time_constraints(max_duration, max_frames, max_samples, exception_expectation):
    # The dummy cuts have a duration of 1 second each
    cut_set = DummyManifest(CutSet, begin_id=0, end_id=100)
    if max_frames is None:
        cut_set = cut_set.drop_features()

    with exception_expectation:
        sampler = SingleCutSampler(
            cut_set,
            shuffle=True,
            # Set an effective batch size of 10 cuts, as all have 1s duration == 100 frames
            # This way we're testing that it works okay when returning multiple batches in
            # a full epoch.
            max_frames=max_frames,
            max_samples=max_samples,
            max_duration=max_duration
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


def test_single_cut_sampler_order_is_deterministic_given_epoch():
    cut_set = DummyManifest(CutSet, begin_id=0, end_id=100)

    sampler = SingleCutSampler(
        cut_set,
        shuffle=True,
        # Set an effective batch size of 10 cuts, as all have 1s duration == 100 frames
        # This way we're testing that it works okay when returning multiple batches in
        # a full epoch.
        max_frames=1000
    )
    sampler.set_epoch(42)
    # calling the sampler twice without epoch update gives identical ordering
    assert [item for item in sampler] == [item for item in sampler]


def test_single_cut_sampler_order_differs_between_epochs():
    cut_set = DummyManifest(CutSet, begin_id=0, end_id=100)

    sampler = SingleCutSampler(
        cut_set,
        shuffle=True,
        # Set an effective batch size of 10 cuts, as all have 1s duration == 100 frames
        # This way we're testing that it works okay when returning multiple batches in
        # a full epoch.
        max_frames=1000
    )
    last_order = [item for item in sampler]
    for epoch in range(1, 6):
        sampler.set_epoch(epoch)
        new_order = [item for item in sampler]
        assert new_order != last_order
        last_order = new_order


def test_single_cut_sampler_len():
    # total duration is 55 seconds
    # each second has 100 frames
    cuts = CutSet.from_cuts(
        dummy_cut(idx, duration=float(idx))
        for idx in range(1, 11)
    )
    sampler = SingleCutSampler(
        cuts,
        shuffle=True,
        max_frames=10 * 100,
        max_cuts=6
    )

    for epoch in range(5):
        assert len(sampler) == len([batch for batch in sampler])
        sampler.set_epoch(epoch)


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
    cut_set = CutSet.from_cuts([
        dummy_cut(0, duration=10),
        dummy_cut(1, duration=20),
    ])
    sampler = CutPairsSampler(
        source_cuts=cut_set,
        target_cuts=cut_set,
        max_source_duration=50,
        max_target_duration=50,
    )
    batch = next(iter(sampler))
    assert len(batch) == 2


@pytest.mark.parametrize(
    ['max_duration', 'max_frames', 'max_samples', 'exception_expectation'],
    [
        (None, None, None, does_not_raise()),  # represents no criterion (unlimited batch size)
        (10.0, None, None, does_not_raise()),
        (None, 1000, None, does_not_raise()),
        (None, None, 160000, does_not_raise()),
        (None, 1000, 160000, pytest.raises(AssertionError)),
        (5.0, 1000, 160000, pytest.raises(AssertionError)),
    ]
)
def test_cut_pairs_sampler_time_constraints(max_duration, max_frames, max_samples, exception_expectation):
    # The dummy cuts have a duration of 1 second each
    cut_set = DummyManifest(CutSet, begin_id=0, end_id=100)
    if max_frames is None:
        cut_set = cut_set.drop_features()

    with exception_expectation:
        sampler = CutPairsSampler(
            source_cuts=cut_set,
            target_cuts=cut_set,
            shuffle=True,
            # Set an effective batch size of 10 cuts, as all have 1s duration == 100 frames
            # This way we're testing that it works okay when returning multiple batches in
            # a full epoch.
            max_source_frames=max_frames,
            max_target_frames=max_frames / 2 if max_frames is not None else None,
            max_source_samples=max_samples,
            max_target_samples=max_samples / 2 if max_samples is not None else None,
            max_source_duration=max_duration,
            max_target_duration=max_duration / 2 if max_duration is not None else None,
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
        max_source_frames=1000,
        max_target_frames=500,
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
        max_source_frames=1000,
        max_target_frames=500,
    )

    last_order = [item for item in sampler]
    for epoch in range(1, 6):
        sampler.set_epoch(epoch)
        new_order = [item for item in sampler]
        assert new_order != last_order
        last_order = new_order


def test_cut_pairs_sampler_len():
    # total duration is 55 seconds
    # each second has 100 frames
    cuts = CutSet.from_cuts(
        dummy_cut(idx, duration=float(idx))
        for idx in range(1, 11)
    )
    sampler = CutPairsSampler(
        source_cuts=cuts,
        target_cuts=cuts,
        shuffle=True,
        max_source_frames=10 * 100,
        max_target_frames=10 * 100,
    )

    for epoch in range(5):
        assert len(sampler) == len([batch for batch in sampler])
        sampler.set_epoch(epoch)


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
    sampler = BucketingSampler(cut_set, sampler_type=SingleCutSampler)
    sampled_cuts = []
    for batch in sampler:
        sampled_cuts.extend(batch)
    assert set(cut_set.ids) == set(c.id for c in sampled_cuts)


def test_bucketing_sampler_shuffle():
    cut_set = DummyManifest(CutSet, begin_id=0, end_id=10)
    sampler = BucketingSampler(cut_set, sampler_type=SingleCutSampler, shuffle=True, num_buckets=2, max_frames=200)

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
    # Internal sampler (SingleCutSampler) ordering should be different in different epochs
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


def test_bucketing_sampler_order_is_deterministic_given_epoch():
    cut_set = DummyManifest(CutSet, begin_id=0, end_id=1000)
    sampler = BucketingSampler(cut_set, sampler_type=SingleCutSampler)

    sampler.set_epoch(42)
    # calling the sampler twice without epoch update gives identical ordering
    assert [item for item in sampler] == [item for item in sampler]


def test_bucketing_sampler_order_differs_between_epochs():
    cut_set = DummyManifest(CutSet, begin_id=0, end_id=1000)
    sampler = BucketingSampler(cut_set, sampler_type=SingleCutSampler)

    last_order = [item for item in sampler]
    for epoch in range(1, 6):
        sampler.set_epoch(epoch)
        new_order = [item for item in sampler]
        assert new_order != last_order
        last_order = new_order


def test_bucketing_sampler_len():
    # total duration is 550 seconds
    # each second has 100 frames
    cuts = CutSet.from_cuts(
        dummy_cut(idx, duration=float(duration))
        for idx, duration in enumerate(list(range(1, 11)) * 10)
    )

    sampler = BucketingSampler(
        cuts,
        num_buckets=4,
        shuffle=True,
        max_frames=64 * 100,
        max_cuts=6
    )

    for epoch in range(5):
        assert len(sampler) == len([item for item in sampler])
        sampler.set_epoch(epoch)


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
    batches = [item for item in sampler]
    assert len(batches) == 15

    # All cuts have the same durations (i.e. are from the same bucket in this case)
    for batch in batches:
        batch_durs = [cut_set[c.id].duration for c in batch]
        assert all(d == batch_durs[0] for d in batch_durs)

    batches = sorted(batches, key=len)
    assert all(len(b) == 1 for b in batches[:10])
    assert all(len(b) == 2 for b in batches[10:])


def test_bucketing_sampler_chooses_buckets_randomly():
    # Construct a CutSet that has 1000 cuts with 100 unique durations.
    # Makes it simple to track which bucket was selected.
    cut_set = CutSet({})  # empty
    for i in range(100):
        new_cuts = DummyManifest(CutSet, begin_id=i * 10, end_id=(i + 1) * 10)
        for c in new_cuts:
            c.duration = i
        cut_set = cut_set + new_cuts

    # Sampler that always select one cut.
    sampler = BucketingSampler(
        cut_set,
        sampler_type=SingleCutSampler,
        max_cuts=1,
        max_frames=1000000000,
        num_buckets=100
    )

    # Batches of 1 guarantee that item is always a single-element list of cut IDs.
    durations = [cut_set[item[0].id].duration for item in sampler]

    # This is the "trick" part - 'groupby' groups the cuts together by their duration.
    # If there is a group that has a size of 10, that means the same bucket was chosen
    # for 10 consecutive batches, which is not what BucketingSampler is supposed to do
    # (the probability of that is extremely low).
    # We're actually setting that threshold lower to 8 which should never be triggered
    # anyway.
    lens = []
    for key, group in groupby(durations):
        lens.append(len(list(group)))
    assert all(l < 8 for l in lens)
    print(lens)


@pytest.mark.parametrize(
    'constraint',
    [
        {'max_frames': 1000},
        {'max_samples': 16000},
        {'max_duration': 10.0}
    ]
)
def test_bucketing_sampler_time_constraints(constraint):
    cut_set = DummyManifest(CutSet, begin_id=0, end_id=1000)
    sampler = BucketingSampler(cut_set, sampler_type=SingleCutSampler, **constraint)
    sampled_cuts = []
    for batch in sampler:
        sampled_cuts.extend(batch)
    assert set(cut_set.ids) == set(c.id for c in sampled_cuts)


@pytest.mark.parametrize('world_size', [2, 3, 4])
@pytest.mark.parametrize('n_cuts', [995, 996, 997, 998, 999, 1000, 1001, 1002, 1003])
@pytest.mark.parametrize('sampler_cls', [SingleCutSampler, BucketingSampler])
def test_partitions_are_equal(world_size, n_cuts, sampler_cls):
    # Create a dummy CutSet.
    cut_set = DummyManifest(CutSet, begin_id=0, end_id=n_cuts)
    # Randomize the durations of cuts to increase the chance we run into edge cases.
    for c in cut_set:
        c.duration += (10 * random.random())
    # Create a sampler for each "distributed worker."
    samplers = [
        sampler_cls(cut_set, max_duration=25.0, rank=i, world_size=world_size)
        for i in range(world_size)
    ]
    # Check that it worked.
    n_batches = [len(s) for s in samplers]
    assert all(nb == n_batches[0] for nb in n_batches)


@pytest.mark.parametrize(
    'sampler_cls',
    [
        SingleCutSampler,
        BucketingSampler,
    ]
)
def test_single_cut_sampler_with_lazy_cuts(sampler_cls):
    # The dummy cuts have a duration of 1 second each
    cut_set = DummyManifest(CutSet, begin_id=0, end_id=100)
    with NamedTemporaryFile(suffix='.jsonl') as f:
        cut_set.to_jsonl(f.name)
        lazy_cuts = CutSet.from_jsonl_lazy(f.name)

        sampler = sampler_cls(
            lazy_cuts,
            shuffle=False,
            # Set an effective batch size of 10 cuts, as all have 1s duration == 100 frames
            # This way we're testing that it works okay when returning multiple batches in
            # a full epoch.
            max_frames=1000
        )
        sampled_cuts = []
        for batch in sampler:
            sampled_cuts.extend(batch)

        # Invariant 1: we receive the same amount of items in a dataloader epoch as there we in the CutSet
        assert len(sampled_cuts) == len(cut_set)
        # Invariant 2: the items are not duplicated
        assert len(set(c.id for c in sampled_cuts)) == len(sampled_cuts)


@pytest.mark.parametrize('sampler_cls', [SingleCutSampler, BucketingSampler])
def test_sampler_filter(sampler_cls):
    # The dummy cuts have a duration of 1 second each
    cut_set = DummyManifest(CutSet, begin_id=0, end_id=100)
    sampler = sampler_cls(
        cut_set,
        shuffle=True,
        # Set an effective batch size of 10 cuts, as all have 1s duration == 100 frames
        # This way we're testing that it works okay when returning multiple batches in
        # a full epoch.
        max_frames=1000
    )
    removed_cut_id = 'dummy-cut-0010'
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
        max_source_frames=1000
    )
    removed_cut_id = 'dummy-cut-0010'
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


def test_zip_sampler():
    cuts1 = DummyManifest(CutSet, begin_id=0, end_id=100)
    cuts2 = DummyManifest(CutSet, begin_id=1000, end_id=1100)
    sampler = ZipSampler(
        # Note: each cut is 1s duration in this test.
        SingleCutSampler(cuts1, max_duration=10),
        SingleCutSampler(cuts2, max_duration=2)
    )
    batches = [b for b in sampler]
    assert len(batches) == 10
    for idx, batch in enumerate(batches):
        assert len(batch) == 12  # twelve 1s items
        assert len([c for c in batch if 0 <= int(c.id.split('-')[-1]) <= 100]) == 10  # ten come from cuts1
        assert len([c for c in batch if 1000 <= int(c.id.split('-')[-1]) <= 1100]) == 2  # two come from cuts2


def test_single_cut_sampler_drop_last():
    # The dummy cuts have a duration of 1 second each
    cut_set = DummyManifest(CutSet, begin_id=0, end_id=100)

    sampler = SingleCutSampler(
        cut_set,
        # Set an effective batch size of 15 cuts, as all have 1s duration == 100 frames
        # This way we're testing that it works okay when returning multiple batches in
        # a full epoch.
        max_frames=1500,
        drop_last=True,
    )
    batches = []
    for batch in sampler:
        assert len(batch) == 15
        batches.append(batch)

    assert len(batches) == 6


@pytest.mark.parametrize('drop_last', [False, True])
def test_bucketing_sampler_drop_last(drop_last):
    # CutSet that has 50 cuts: 10 have 1s, 10 have 2s, etc.
    cut_set = CutSet()
    for i in range(5):
        new_cuts = DummyManifest(CutSet, begin_id=i * 10, end_id=(i + 1) * 10)
        for c in new_cuts:
            c.duration = i + 1
        cut_set = cut_set + new_cuts

    # Sampler that always select one cut.
    sampler = BucketingSampler(
        cut_set,
        sampler_type=SingleCutSampler,
        max_duration=10.5,
        num_buckets=5,
        drop_last=drop_last,
    )
    batches = []
    for batch in sampler:
        # Assert there is a consistent cut duration per bucket in this test.
        for cut in batch:
            assert cut.duration == batch[0].duration
        batches.append(batch)

    # Expectation:
    if drop_last:
        # When drop_last = True:
        #   10 x 1s cuts == 1 batch (10 cuts each, 0 left over)
        #   10 x 2s cuts == 2 batches (5 cuts each, 0 left over)
        #   10 x 3s cuts == 3 batches (3 cuts each, 1 left over)
        #   10 x 4s cuts == 5 batches (2 cuts each, 0 left over)
        #   10 x 5s cuts == 5 batches (2 cuts each, 0 left over)
        expected_num_batches = 16
        expected_num_cuts = 49
        expected_discarded_cuts = 1
    else:
        # When drop_last = False:
        #   There will be one more batch with a single 3s cut.
        expected_num_batches = 17
        expected_num_cuts = 50
        expected_discarded_cuts = 0

    num_sampled_cuts = sum(len(b) for b in batches)
    num_discarded_cuts = len(cut_set) - num_sampled_cuts
    assert len(batches) == expected_num_batches
    assert num_sampled_cuts == expected_num_cuts
    assert num_discarded_cuts == expected_discarded_cuts


@pytest.mark.parametrize(
    "sampler",
    [
        SingleCutSampler(DummyManifest(CutSet, begin_id=0, end_id=10)),
        CutPairsSampler(
            DummyManifest(CutSet, begin_id=0, end_id=10),
            DummyManifest(CutSet, begin_id=0, end_id=10),
        ),
        BucketingSampler(DummyManifest(CutSet, begin_id=0, end_id=10)),
        ZipSampler(
            SingleCutSampler(DummyManifest(CutSet, begin_id=0, end_id=10)),
            SingleCutSampler(DummyManifest(CutSet, begin_id=0, end_id=10)),
        ),
    ],
)
def test_sampler_get_report(sampler):
    _ = [b for b in sampler]
    print(sampler.get_report())
    # It runs - voila!


@pytest.mark.parametrize(
    'sampler_cls',
    [
        SingleCutSampler,
        BucketingSampler,
    ]
)
def test_single_cut_sampler_lazy_shuffle(sampler_cls):
    # The dummy cuts have a duration of 1 second each
    cut_set = DummyManifest(CutSet, begin_id=0, end_id=100)
    with NamedTemporaryFile(suffix='.jsonl') as f:
        cut_set.to_jsonl(f.name)
        lazy_cuts = CutSet.from_jsonl_lazy(f.name)

        sampler = sampler_cls(
            lazy_cuts,
            shuffle=True,
            # Set an effective batch size of 10 cuts, as all have 1s duration == 100 frames
            # This way we're testing that it works okay when returning multiple batches in
            # a full epoch.
            max_frames=1000
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
    'sampler_cls',
    [
        CutPairsSampler,
        pytest.param(
            BucketingSampler,
            marks=pytest.mark.xfail(reason="BucketingSampler does not support lazy cuts pairs yet."),
        )
    ]
)
def test_cut_pairs_sampler_lazy_shuffle(sampler_cls):
    # The dummy cuts have a duration of 1 second each
    cut_set = DummyManifest(CutSet, begin_id=0, end_id=100)
    with NamedTemporaryFile(suffix='.jsonl') as f:
        cut_set.to_jsonl(f.name)
        lazy_cuts = CutSet.from_jsonl_lazy(f.name)

        sampler = sampler_cls(
            lazy_cuts,
            lazy_cuts,
            shuffle=True,
            # Set an effective batch size of 10 cuts, as all have 1s duration == 100 frames
            # This way we're testing that it works okay when returning multiple batches in
            # a full epoch.
            max_source_frames=1000
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


@pytest.mark.parametrize('datasize', [10, 1000, 20000])
@pytest.mark.parametrize('bufsize', [100, 1000, 10000])
def test_streaming_shuffle(datasize, bufsize):
    data = list(range(int(datasize)))
    shuffled = list(streaming_shuffle(iter(data), bufsize=int(bufsize)))
    assert len(data) == len(shuffled)
    assert len(shuffled) == len(set(shuffled))
    assert data != shuffled
