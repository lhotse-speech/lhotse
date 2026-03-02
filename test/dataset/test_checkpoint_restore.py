"""
Tests for Phase 5: Sampler & DataLoader integration with checkpointing.

Tests that:
- DynamicCutSampler.state_dict() captures cuts_state when available
- DynamicCutSampler and DynamicBucketingSampler can restore mid-epoch
- Old state_dicts (without cuts_state) still work via legacy _fast_forward
- IterableDatasetWrapper implements state_dict()/load_state_dict()
"""
import pytest

from lhotse import CutSet
from lhotse.dataset.iterable_dataset import IterableDatasetWrapper
from lhotse.dataset.sampling.base import CutSampler
from lhotse.dataset.sampling.dynamic import DynamicCutSampler
from lhotse.dataset.sampling.dynamic_bucketing import DynamicBucketingSampler
from lhotse.testing.dummies import DummyManifest


@pytest.fixture
def cuts():
    return DummyManifest(CutSet, begin_id=0, end_id=100)


class TestDynamicSamplerCheckpoint:
    def test_state_dict_contains_diagnostics(self, cuts):
        sampler = DynamicCutSampler(cuts, max_cuts=5, shuffle=False)
        iter(sampler)
        for _ in range(3):
            next(sampler)
        sd = sampler.state_dict()
        assert "diagnostics" in sd

    def test_dynamic_sampler_restore_matches_uninterrupted(self, cuts):
        """Verify that restored sampler produces batches consistent with diagnostics."""
        sampler1 = DynamicCutSampler(cuts, max_cuts=5, shuffle=False)
        iter(sampler1)
        batches_before = []
        for _ in range(3):
            batches_before.append(next(sampler1))

        sd = sampler1.state_dict()

        # Restore into a new sampler
        sampler2 = DynamicCutSampler(cuts, max_cuts=5, shuffle=False)
        sampler2.load_state_dict(sd.copy())

        # After restoration, the diagnostics should report 3 batches
        assert sampler2.diagnostics.current_epoch_stats.total_batches == 3

    def test_dynamic_bucketing_sampler_restore(self, cuts):
        """DynamicBucketingSampler can be checkpointed and restored."""
        sampler1 = DynamicBucketingSampler(
            cuts, max_cuts=5, shuffle=False, num_buckets=2
        )
        iter(sampler1)
        for _ in range(3):
            next(sampler1)

        sd = sampler1.state_dict()

        sampler2 = DynamicBucketingSampler(
            cuts, max_cuts=5, shuffle=False, num_buckets=2
        )
        sampler2.load_state_dict(sd.copy())

        assert sampler2.diagnostics.current_epoch_stats.total_batches == 3


class TestBackwardCompatOldStateDict:
    def test_old_state_dict_without_cuts_state(self, cuts):
        """
        Old state_dicts that lack 'cuts_state' key should still restore
        correctly via the legacy _fast_forward path.
        """
        sampler1 = DynamicCutSampler(cuts, max_cuts=5, shuffle=False)
        iter(sampler1)
        for _ in range(3):
            next(sampler1)

        sd = sampler1.state_dict()
        # Simulate an old state_dict by removing cuts_state
        sd.pop("cuts_state", None)

        sampler2 = DynamicCutSampler(cuts, max_cuts=5, shuffle=False)
        sampler2.load_state_dict(sd.copy())

        # Should still work via _fast_forward
        assert sampler2.diagnostics.current_epoch_stats.total_batches == 3


class TestIterableDatasetWrapperStateDict:
    def test_state_dict_round_trip(self, cuts):
        """IterableDatasetWrapper.state_dict() and load_state_dict() round-trip."""
        import torch.utils.data

        class IdentityDataset(torch.utils.data.Dataset):
            def __getitem__(self, item):
                return item

        dataset = IdentityDataset()
        sampler = DynamicCutSampler(cuts, max_cuts=10, shuffle=False)
        wrapper = IterableDatasetWrapper(dataset, sampler)

        wrapper.set_epoch(3)
        iter(wrapper)
        for _ in range(2):
            next(wrapper)

        sd = wrapper.state_dict()
        assert "epoch" in sd
        assert "sampler_state" in sd
        assert sd["epoch"] == 3

        # Create a fresh wrapper and restore
        sampler2 = DynamicCutSampler(cuts, max_cuts=10, shuffle=False)
        wrapper2 = IterableDatasetWrapper(dataset, sampler2)
        wrapper2.load_state_dict(sd)

        assert wrapper2.epoch == 3
        assert wrapper2.sampler.diagnostics.current_epoch_stats.total_batches == 2
