import pytest

from lhotse import CutSet
from lhotse.dataset.sampling import SingleCutSampler
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
    batches = []
    for batch in sampler:
        batches.append(batch)
        sampler_cut_ids.extend(c.id for c in batch)

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
