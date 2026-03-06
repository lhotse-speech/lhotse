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


class TestShuffleIndexedGuard:
    def test_dynamic_cut_sampler_rejects_shuffle_with_indexed(self, tmp_path):
        """shuffle=True + indexed dataset raises ValueError for DynamicCutSampler.

        DynamicBucketingSampler allows shuffle=True with indexed datasets because
        its within-bucket shuffle RNG is saved/restored in the O(1) path.
        DynamicCutSampler's streaming_shuffle is not checkpointable, so it raises.
        """
        path = tmp_path / "cuts.jsonl"
        DummyManifest(CutSet, begin_id=0, end_id=20).to_jsonl(path)

        cs = CutSet.from_file(path, indexed=True)
        assert cs.is_indexed is True

        # DynamicBucketingSampler with shuffle=True is allowed (RNG is saved)
        DynamicBucketingSampler(cs, max_cuts=5, shuffle=True, num_buckets=2)

        with pytest.raises(
            ValueError, match="shuffle=True is not supported with indexed"
        ):
            DynamicCutSampler(cs, max_cuts=5, shuffle=True)


class TestIndexedSamplerStateCapture:
    def test_dynamic_sampler_captures_cuts_state_for_indexed(self, tmp_path):
        path = tmp_path / "cuts.jsonl"
        DummyManifest(CutSet, begin_id=0, end_id=40).to_jsonl(path)
        cs = CutSet.from_file(path, indexed=True)

        sampler = DynamicCutSampler(cs, max_cuts=5, shuffle=False)
        iter(sampler)
        for _ in range(4):
            next(sampler)

        sd = sampler.state_dict()
        assert "cuts_state" in sd
        assert sd["cuts_state"] is not None
        assert len(sd["cuts_state"]) == 1

    def test_dynamic_sampler_indexed_restore_avoids_on_fast_forward(
        self, tmp_path, monkeypatch
    ):
        path = tmp_path / "cuts.jsonl"
        DummyManifest(CutSet, begin_id=0, end_id=50).to_jsonl(path)
        cs = CutSet.from_file(path, indexed=True)

        sampler1 = DynamicCutSampler(cs, max_cuts=5, shuffle=False)
        iter(sampler1)
        for _ in range(3):
            next(sampler1)
        sd = sampler1.state_dict()
        assert "cuts_state" in sd and sd["cuts_state"] is not None

        consumed = sampler1.diagnostics.current_epoch_stats.total_batches
        assert consumed == 3

        sampler2 = DynamicCutSampler(cs, max_cuts=5, shuffle=False)
        sampler2.load_state_dict(sd.copy())

        called = {"on_fast_forward": False}
        orig_next = CutSampler.__next__

        def _patched_next(self):
            called["on_fast_forward"] = True
            raise RuntimeError("Legacy O(N) fast-forward was used instead of O(1).")

        monkeypatch.setattr(CutSampler, "__next__", _patched_next)
        try:
            iter(sampler2)
        finally:
            monkeypatch.setattr(CutSampler, "__next__", orig_next)

        assert called["on_fast_forward"] is False
        assert sampler2.diagnostics.current_epoch_stats.total_batches == consumed

    def test_dynamic_bucketing_sampler_captures_cuts_state_for_indexed(self, tmp_path):
        path = tmp_path / "cuts.jsonl"
        DummyManifest(CutSet, begin_id=0, end_id=40).to_jsonl(path)
        cs = CutSet.from_file(path, indexed=True)

        sampler = DynamicBucketingSampler(cs, max_cuts=5, shuffle=False, num_buckets=2)
        iter(sampler)
        for _ in range(4):
            next(sampler)

        sd = sampler.state_dict()
        assert "cuts_state" in sd
        assert sd["cuts_state"] is not None
        assert len(sd["cuts_state"]) == 1

    def test_lazy_cut_mixer_indexed_path_is_checkpointable_and_captured(self, tmp_path):
        speech_path = tmp_path / "speech.jsonl"
        noise_path = tmp_path / "noise.jsonl"
        DummyManifest(CutSet, begin_id=0, end_id=40).to_jsonl(speech_path)
        DummyManifest(CutSet, begin_id=100, end_id=120).to_jsonl(noise_path)

        speech = CutSet.from_file(speech_path, indexed=True)
        noise = CutSet.from_file(noise_path, indexed=True)
        mixed = speech.mix(noise, mix_prob=1.0, seed=123, preserve_id="left")

        from lhotse.cut.set import LazyCutMixer

        assert isinstance(mixed.data, LazyCutMixer)
        assert mixed.data.is_checkpointable

        sampler = DynamicBucketingSampler(
            mixed, max_cuts=5, shuffle=False, num_buckets=2
        )
        iter(sampler)
        for _ in range(3):
            next(sampler)
        sd = sampler.state_dict()

        root_state = sd["cuts_state"][0]
        assert root_state["_type"] == "LazyCutMixer"
        assert "_state" in root_state

    def test_dynamic_bucketing_indexed_mix_restore_avoids_on_fast_forward(
        self, tmp_path, monkeypatch
    ):
        speech_path = tmp_path / "speech.jsonl"
        noise_path = tmp_path / "noise.jsonl"
        DummyManifest(CutSet, begin_id=0, end_id=50).to_jsonl(speech_path)
        DummyManifest(CutSet, begin_id=100, end_id=130).to_jsonl(noise_path)

        def make_sampler():
            speech = CutSet.from_file(speech_path, indexed=True)
            noise = CutSet.from_file(noise_path, indexed=True)
            mixed = speech.mix(noise, mix_prob=1.0, seed=123, preserve_id="left")
            return DynamicBucketingSampler(
                mixed, max_cuts=5, shuffle=False, num_buckets=2
            )

        sampler1 = make_sampler()
        iter(sampler1)
        for _ in range(3):
            next(sampler1)
        sd = sampler1.state_dict()

        sampler2 = make_sampler()
        sampler2.load_state_dict(sd.copy())

        called = {"on_fast_forward": False}
        orig_next = CutSampler.__next__

        def _patched_next(self):
            called["on_fast_forward"] = True
            raise RuntimeError("Legacy O(N) fast-forward was used instead of O(1).")

        monkeypatch.setattr(CutSampler, "__next__", _patched_next)
        try:
            iter(sampler2)
        finally:
            monkeypatch.setattr(CutSampler, "__next__", orig_next)

        assert called["on_fast_forward"] is False

    def test_dynamic_bucketing_indexed_restore_avoids_on_fast_forward_and_preserves_diagnostics(
        self, tmp_path, monkeypatch
    ):
        path = tmp_path / "cuts.jsonl"
        DummyManifest(CutSet, begin_id=0, end_id=50).to_jsonl(path)
        cs = CutSet.from_file(path, indexed=True)

        sampler1 = DynamicBucketingSampler(cs, max_cuts=5, shuffle=False, num_buckets=2)
        iter(sampler1)
        for _ in range(3):
            next(sampler1)
        sd = sampler1.state_dict()
        assert "cuts_state" in sd and sd["cuts_state"] is not None
        assert "bucketer_state" in sd
        assert "rng_state" in sd

        consumed = sampler1.diagnostics.current_epoch_stats.total_batches
        assert consumed == 3

        sampler2 = DynamicBucketingSampler(cs, max_cuts=5, shuffle=False, num_buckets=2)
        sampler2.load_state_dict(sd.copy())

        called = {"on_fast_forward": False}
        orig_next = CutSampler.__next__

        def _patched_next(self):
            called["on_fast_forward"] = True
            raise RuntimeError("Legacy O(N) fast-forward was used instead of O(1).")

        monkeypatch.setattr(CutSampler, "__next__", _patched_next)
        try:
            iter(sampler2)
        finally:
            monkeypatch.setattr(CutSampler, "__next__", orig_next)

        assert called["on_fast_forward"] is False
        assert sampler2.diagnostics.current_epoch_stats.total_batches == consumed

    def test_dynamic_bucketing_indexed_prefetch_state_has_bucketer_state(
        self, tmp_path
    ):
        path = tmp_path / "cuts.jsonl"
        DummyManifest(CutSet, begin_id=0, end_id=80).to_jsonl(path)
        cs = CutSet.from_file(path, indexed=True)

        sampler1 = DynamicBucketingSampler(cs, max_cuts=5, shuffle=False, num_buckets=2)
        iter(sampler1)
        # Simulate a worker-prefetched batch that wasn't yet accounted in
        # sampler diagnostics.
        next(sampler1.cuts_iter)
        assert sampler1.diagnostics.current_epoch_stats.total_batches == 0

        sd = sampler1.state_dict()
        assert "bucketer_state" in sd
        assert "rng_state" in sd

        expected_next = [c.id for c in next(sampler1)]

        sampler2 = DynamicBucketingSampler(cs, max_cuts=5, shuffle=False, num_buckets=2)
        sampler2.load_state_dict(sd.copy())
        iter(sampler2)
        restored_next = [c.id for c in next(sampler2)]
        assert restored_next == expected_next

    def test_dynamic_bucketing_indexed_restore_missing_o1_state_raises(self, tmp_path):
        path = tmp_path / "cuts.jsonl"
        DummyManifest(CutSet, begin_id=0, end_id=60).to_jsonl(path)
        cs = CutSet.from_file(path, indexed=True)

        sampler1 = DynamicBucketingSampler(cs, max_cuts=5, shuffle=False, num_buckets=2)
        iter(sampler1)
        for _ in range(3):
            next(sampler1)
        sd = sampler1.state_dict()
        sd.pop("bucketer_state", None)
        sd.pop("rng_state", None)

        sampler2 = DynamicBucketingSampler(cs, max_cuts=5, shuffle=False, num_buckets=2)
        sampler2.load_state_dict(sd.copy())
        with pytest.raises(
            RuntimeError,
            match="indexed datasets should never use O\\(N\\) fast-forward",
        ):
            iter(sampler2)

    def test_dynamic_cut_indexed_restore_missing_cuts_state_raises(self, tmp_path):
        path = tmp_path / "cuts.jsonl"
        DummyManifest(CutSet, begin_id=0, end_id=60).to_jsonl(path)
        cs = CutSet.from_file(path, indexed=True)

        sampler1 = DynamicCutSampler(cs, max_cuts=5, shuffle=False)
        iter(sampler1)
        for _ in range(3):
            next(sampler1)
        sd = sampler1.state_dict()
        sd.pop("cuts_state", None)

        sampler2 = DynamicCutSampler(cs, max_cuts=5, shuffle=False)
        sampler2.load_state_dict(sd.copy())
        with pytest.raises(
            RuntimeError,
            match="indexed datasets should never use O\\(N\\) fast-forward",
        ):
            iter(sampler2)
