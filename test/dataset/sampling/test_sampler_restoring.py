from copy import deepcopy
from tempfile import NamedTemporaryFile
from typing import List

import pytest
import torch
from torch.utils.data import DataLoader

from lhotse import CutSet
from lhotse.dataset import (
    BucketingSampler,
    CutPairsSampler,
    DynamicBucketingSampler,
    RoundRobinSampler,
    SimpleCutSampler,
    ZipSampler,
)
from lhotse.dataset.sampling.dynamic import DynamicCutSampler
from lhotse.testing.dummies import DummyManifest

CUTS = DummyManifest(CutSet, begin_id=0, end_id=100)
CUTS_MOD = CUTS.modify_ids(lambda cid: cid + "_alt")

# fmt: off
SAMPLERS_TO_TEST = [
    # Identically initialized SimpleCutSampler
    lambda: (
        SimpleCutSampler(CUTS, max_duration=10.0, shuffle=True, drop_last=True),
        SimpleCutSampler(CUTS, max_duration=10.0, shuffle=True, drop_last=True),
    ),
    # Differently initialized SimpleCutSampler with the same CUTS
    lambda: (
        SimpleCutSampler(CUTS, max_duration=10.0, shuffle=True, drop_last=True),
        SimpleCutSampler(CUTS, max_duration=11.1),
    ),
    # Differently initialized CutPairsSampler with the same CUTS
    lambda: (
        CutPairsSampler(CUTS, CUTS, max_source_duration=10.0, shuffle=True, drop_last=True),
        CutPairsSampler(CUTS, CUTS),
    ),
    # Differently initialized ZipSampler with the same CUTS
    lambda: (
        ZipSampler(  # CUTS_SHUF just to randomize the order of the zipped-in cutset
            SimpleCutSampler(CUTS, max_duration=10.0, shuffle=True, drop_last=True),
            SimpleCutSampler(CUTS_MOD, max_duration=10.0, shuffle=True, drop_last=True),
        ),
        ZipSampler(
            SimpleCutSampler(CUTS, max_duration=11.1),
            SimpleCutSampler(CUTS_MOD, max_duration=11.1),
        ),
    ),
    # Differently initialized ZipSampler with the same CUTS (cut pairs)
    lambda: (
        ZipSampler(
            CutPairsSampler(CUTS, CUTS, max_source_duration=10.0, shuffle=True, drop_last=True),
            CutPairsSampler(CUTS_MOD, CUTS_MOD, max_source_duration=10.0, shuffle=True, drop_last=True),
        ),
        ZipSampler(
            CutPairsSampler(CUTS, CUTS),
            CutPairsSampler(CUTS_MOD, CUTS_MOD),
        ),
    ),
    # Differently initialized BucketingSampler with the same CUTS
    lambda: (
        BucketingSampler(CUTS, max_duration=10.0, shuffle=True, drop_last=True, num_buckets=2),
        BucketingSampler(CUTS, max_duration=11.1, num_buckets=2),
    ),
    # Differently initialized BucketingSampler (using CutPairsSampler) with the same CUTS
    lambda: (
        BucketingSampler(CUTS, CUTS, max_source_duration=10.0, shuffle=True, drop_last=True, num_buckets=2,
                         sampler_type=CutPairsSampler),
        BucketingSampler(CUTS, CUTS, num_buckets=2, sampler_type=CutPairsSampler),
    ),
    lambda: (
        DynamicBucketingSampler(CUTS, max_duration=10.0, shuffle=True, drop_last=True, num_buckets=2),
        DynamicBucketingSampler(CUTS, max_duration=10.0, num_buckets=2),
    ),
    lambda: (
        DynamicCutSampler(CUTS, max_duration=10.0, shuffle=True, drop_last=True),
        DynamicCutSampler(CUTS, max_duration=10.0),
    ),
    # Differently initialized RoundRobinSampler with the same CUTS
    lambda: (
        RoundRobinSampler(
            SimpleCutSampler(CUTS.subset(first=50), max_duration=10.0, shuffle=True, drop_last=True),
            SimpleCutSampler(CUTS_MOD.subset(first=50), max_duration=10.0, shuffle=True, drop_last=True),
        ),
        RoundRobinSampler(
            SimpleCutSampler(CUTS.subset(first=50), max_duration=11.1),
            SimpleCutSampler(CUTS_MOD.subset(first=50), max_duration=11.1),
        ),
    ),
]
# fmt: on


@pytest.mark.parametrize("create_samplers", SAMPLERS_TO_TEST)
def test_restore_sampler_state(create_samplers):
    sampler, restored_sampler = create_samplers()
    # Iterate a full epoch through the sampler first to accumulate some sampling diagnostics.
    sampler.set_epoch(0)
    list(sampler)
    sampler.set_epoch(1)
    list(sampler)
    sampler.set_epoch(2)
    list(sampler)

    # Iterate the sampler a bit. With max_duration=10s, all samplers should have 10 batches total per epoch.
    sampler.set_epoch(3)
    iter(sampler)
    for _ in range(5):
        next(sampler)

    # Restore state.
    state_dict = sampler.state_dict()
    # Call .state_dict() again, because load_state_dict() is mutating the input.
    restored_sampler.load_state_dict(sampler.state_dict())

    # Check it's the same: state dicts.
    # They have the same keys.
    restored_state_dict = restored_sampler.state_dict()
    assert set(state_dict) & set(restored_state_dict) == set(state_dict) | set(
        restored_state_dict
    )
    # They have the same values.
    for key in state_dict:
        assert state_dict[key] == restored_state_dict[key]

    # Check that both yield the same batches until the end of the epoch.
    orig_batches = []
    while True:
        # Careful: `for batch in sampler` would trigger iter(), resetting the iteration state.
        try:
            orig_batches.append(next(sampler))
        except StopIteration:
            break

    restored_batches = []
    for b in restored_sampler:
        # Iterate the restored sampler normally, as it is intended to be used this way.
        # It "knows" that it was just restored and won't reset it's state on the first call to iter().
        restored_batches.append(b)

    # There should be a total of 5 batches since we're starting mid-epoch.
    assert len(orig_batches) == 5
    assert len(restored_batches) == 5

    assert orig_batches[0] == restored_batches[0]
    assert orig_batches[1] == restored_batches[1]
    assert orig_batches[2] == restored_batches[2]
    assert orig_batches[3] == restored_batches[3]
    assert orig_batches[4] == restored_batches[4]


@pytest.mark.parametrize("create_samplers", SAMPLERS_TO_TEST)
def test_restored_sampler_continues_as_normal(create_samplers):
    sampler, restored_sampler = create_samplers()

    # Iterate the sampler a bit. With max_duration=10s, all samplers should have 10 batches total per epoch.
    sampler.set_epoch(3)
    iter(sampler)
    for _ in range(5):
        next(sampler)

    # Restore state.
    restored_sampler.load_state_dict(sampler.state_dict())

    batches = []
    for b in restored_sampler:
        # Iterate the restored sampler normally, as it is intended to be used this way.
        # It "knows" that it was just restored and won't reset it's state on the first call to iter().
        batches.append(b)

    # There should be a total of 5 batches since we're starting mid-epoch.
    assert len(batches) == 5

    # Now check that when we iterate the same epoch again, there are full 10 batches.
    batches_reiter = []
    for b in restored_sampler:
        batches_reiter.append(b)
    assert len(batches_reiter) == 10


@pytest.mark.parametrize("create_samplers", SAMPLERS_TO_TEST)
def test_restored_sampler_forced_to_start_from_scratch(create_samplers):
    sampler, restored_sampler = create_samplers()

    # Iterate the sampler a bit. With max_duration=10s, all samplers should have 10 batches total per epoch.
    sampler.set_epoch(3)
    iter(sampler)
    for _ in range(5):
        next(sampler)

    # Restore state.
    restored_sampler.load_state_dict(sampler.state_dict())

    # Tells the sampler that we want to discard the current progress.
    restored_sampler.allow_iter_to_reset_state()

    batches = []
    for b in restored_sampler:
        # Iterate the restored sampler normally, as it is intended to be used this way.
        # It "knows" that it was just restored and won't reset it's state on the first call to iter().
        batches.append(b)

    # There should be a total of 10 batches since we're starting mid-epoch.
    assert len(batches) == 10


@pytest.mark.parametrize("create_samplers", SAMPLERS_TO_TEST)
def test_save_and_load_sampler_state(create_samplers):
    sampler, restored_sampler = create_samplers()
    sd = sampler.state_dict()
    with NamedTemporaryFile(suffix=".pt") as f:
        torch.save(sd, f.name)
        f.flush()
        sd_restored = torch.load(f.name)
    assert sd == sd_restored


class DummyDataset(torch.utils.data.Dataset):
    def __getitem__(self, item: CutSet) -> List[str]:
        return list(item.ids)


@pytest.mark.parametrize(
    "create_sampler",
    [
        lambda: SimpleCutSampler(CUTS, max_duration=10.0, shuffle=True, drop_last=True),
        lambda: BucketingSampler(
            CUTS, max_duration=10.0, shuffle=True, drop_last=True, num_buckets=2
        ),
        lambda: ZipSampler(
            SimpleCutSampler(CUTS, max_duration=10.0, shuffle=True, drop_last=True),
            SimpleCutSampler(CUTS_MOD, max_duration=10.0, shuffle=True, drop_last=True),
        ),
        lambda: RoundRobinSampler(
            # subset to first 50 cuts so that total num batches is 10, not 20
            SimpleCutSampler(
                CUTS.subset(first=50), max_duration=10.0, shuffle=True, drop_last=True
            ),
            SimpleCutSampler(
                CUTS_MOD.subset(first=50),
                max_duration=10.0,
                shuffle=True,
                drop_last=True,
            ),
        ),
        lambda: DynamicCutSampler(
            CUTS, max_duration=10.0, shuffle=True, drop_last=True
        ),
        lambda: DynamicBucketingSampler(
            CUTS, max_duration=10.0, shuffle=True, drop_last=True
        ),
    ],
)
@pytest.mark.parametrize("num_workers", [0, 1])
def test_e2e_restore_with_dataloader(num_workers, create_sampler):
    sampler = create_sampler()
    restored_sampler = deepcopy(
        sampler
    )  # keep fresh sampler for state restoration later
    dset = DummyDataset()
    # Expecting 10 batches in total.
    # Note: not testing with num_workers > 1 as it will randomize the order of batches.
    dloader = DataLoader(
        dset, batch_size=None, sampler=sampler, num_workers=num_workers
    )

    # Iterate the dataloader for one epoch to accumulate some stats.
    dloader.sampler.set_epoch(0)
    list(dloader)

    # Advance to epoch 1 and iterate it partially to check resuming.
    dloader.sampler.set_epoch(1)
    expected_batches = []
    for idx, b in enumerate(dloader):
        if idx == 4:
            # Save the training loop state at step 4 to resume from later.
            state = dloader.sampler.state_dict()
        if idx > 4:
            # Continue iterating to see what batches should be sampled after restoring.
            expected_batches.append(b)

    # Restore the sampler to its state from the dloader.
    restored_sampler.load_state_dict(state)

    # Initialize a new dloader with the restored sampler.
    restored_dloader = DataLoader(
        dset, batch_size=None, sampler=restored_sampler, num_workers=num_workers
    )
    batches = []
    for b in restored_dloader:
        batches.append(b)

    # Check that the results are the same.
    assert len(expected_batches) == 5
    if num_workers == 0:
        assert len(batches) == 5
        assert batches == expected_batches
    else:
        # We "lost" 2 batches due to prefetching (i.e., the sampler's state was ahead by 2 batches
        # and we cannot recover from it for now)
        assert len(batches) == 3
        assert batches == expected_batches[2:]

    # Advance the epoch and make sure it is fully iterated
    restored_dloader.sampler.set_epoch(2)
    ep2_batches = list(restored_dloader)
    assert len(ep2_batches) == 10


@pytest.mark.parametrize("create_samplers", SAMPLERS_TO_TEST)
@pytest.mark.parametrize("advance_epoch", [True, False])
def test_epoch_diagnostics_reset_on_start_of_iteration(create_samplers, advance_epoch):
    # We only need one sampler for this test to check that it properly
    # resets the cut/batch counter if we re-iterate over the same epoch
    sampler, _ = create_samplers()

    assert sampler.diagnostics.current_epoch == 0
    assert sampler.diagnostics.total_cuts == 0
    assert sampler.diagnostics.total_batches == 0

    # Iterate a single epoch
    for _ in sampler:
        pass

    assert sampler.diagnostics.current_epoch == 0
    total_cuts_pass1 = sampler.diagnostics.total_cuts
    total_batches_pass1 = sampler.diagnostics.total_batches

    if advance_epoch:
        sampler.set_epoch(1)

    # if advance_epoch:
    #   Iterate the next epoch, which should increment the next epoch stats
    #   We expect the total sampler stats to be incremented
    # else:
    #   Iterate a single epoch again, which should reset the current stats
    #   We expect the total sampler stats to be identical
    for _ in sampler:
        pass

    if advance_epoch:
        assert sampler.diagnostics.current_epoch == 1
        assert sampler.diagnostics.total_cuts > total_cuts_pass1
        assert sampler.diagnostics.total_batches > total_batches_pass1
    else:
        assert sampler.diagnostics.current_epoch == 0
        assert sampler.diagnostics.total_cuts == total_cuts_pass1
        assert sampler.diagnostics.total_batches == total_batches_pass1
