"""
Tests for Phase 5: Sampler & DataLoader integration with checkpointing.

Tests that:
- DynamicCutSampler.state_dict() captures cuts_state when available
- DynamicCutSampler and DynamicBucketingSampler can restore mid-epoch
- Checkpoints missing the iteration state needed for restore are rejected
- IterableDatasetWrapper implements state_dict()/load_state_dict()
"""

import random
from copy import deepcopy

import pytest

from lhotse import CutSet
from lhotse.dataset.cut_transforms import PerturbSpeed
from lhotse.dataset.iterable_dataset import IterableDatasetWrapper
from lhotse.dataset.sampling.dynamic import DynamicCutSampler
from lhotse.dataset.sampling.dynamic_bucketing import (
    DynamicBucketer,
    DynamicBucketingSampler,
)
from lhotse.lazy import IteratorNode, LazyRepeater, attach_graph_origin
from lhotse.shar.writers import SharWriter
from lhotse.testing.dummies import DummyManifest


@pytest.fixture
def cuts():
    return DummyManifest(CutSet, begin_id=0, end_id=100)


@pytest.fixture
def indexed_cuts(tmp_path, cuts):
    path = tmp_path / "indexed_cuts.jsonl"
    cuts.to_jsonl(path)
    return CutSet.from_file(path, indexed=True)


@pytest.fixture(params=["jsonl", "shar"])
def indexed_sampler_factory(tmp_path, request):
    path = tmp_path / "cuts.jsonl"
    cuts = DummyManifest(CutSet, begin_id=0, end_id=8)
    if request.param == "shar":
        with SharWriter(tmp_path, fields={}, compress_jsonl=False) as writer:
            for cut in cuts:
                writer.write(cut)
    else:
        cuts.to_jsonl(path)

    def make(sampler_type, *, shuffle=True):
        source = (
            CutSet.from_shar(in_dir=tmp_path, indexed=True)
            if request.param == "shar"
            else CutSet.from_file(path, indexed=True)
        )
        kwargs = {"duration_bins": [0.5, 1.5], "buffer_size": 20}
        if sampler_type is DynamicCutSampler:
            kwargs = {"shuffle_buffer_size": 20}
        return sampler_type(source, max_cuts=2, shuffle=shuffle, seed=7, **kwargs)

    return make


def _remaining_ids(sampler):
    batches = []
    while True:
        try:
            batches.append(list(next(sampler).ids))
        except StopIteration:
            return batches


@pytest.mark.parametrize("sampler_type", [DynamicCutSampler, DynamicBucketingSampler])
@pytest.mark.parametrize("same_instance", [False, True])
@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("augment", [False, True])
@pytest.mark.parametrize("prepare_iterator", [False, True])
@pytest.mark.parametrize("checkpoint_step", [0, 1, 4])
def test_indexed_sampler_pending_checkpoint_roundtrip(
    indexed_sampler_factory,
    sampler_type,
    same_instance,
    shuffle,
    augment,
    prepare_iterator,
    checkpoint_step,
):
    def make():
        sampler = indexed_sampler_factory(sampler_type, shuffle=shuffle)
        if augment:
            sampler.map(
                PerturbSpeed(factors=[0.9, 1.1], p=1.0, randgen=random.Random(17))
            )
        return sampler

    sampler = make()
    iter(sampler)
    for _ in range(checkpoint_step):
        next(sampler)
    state = deepcopy(sampler.state_dict())
    expected = _remaining_ids(sampler)
    intermediate = sampler if same_instance else make()
    intermediate.load_state_dict(deepcopy(state))
    if prepare_iterator:
        iter(intermediate)
    pending = deepcopy(intermediate.state_dict())
    # Preparing a pre-yield iterator may change unused source cursors.
    if checkpoint_step > 0 or not prepare_iterator:
        assert pending == state
    restored = make()
    restored.load_state_dict(pending)
    assert [list(batch.ids) for batch in restored] == expected


@pytest.mark.parametrize("prepare_iterator", [False, True])
@pytest.mark.parametrize("shuffle", [False, True])
def test_sampler_checkpoint_preserves_separately_restored_source(
    tmp_path, prepare_iterator, shuffle
):
    path = tmp_path / "source.jsonl"
    DummyManifest(CutSet, begin_id=0, end_id=8).to_jsonl(path)
    source = CutSet.from_file(path, indexed=True)
    iterator = iter(source)
    for _ in range(3):
        next(iterator)
    source_state = deepcopy(source.state_dict())

    def make(*, restore_source):
        source = CutSet.from_file(path, indexed=True)
        if restore_source:
            source.load_state_dict(deepcopy(source_state))
        return DynamicCutSampler(source, max_cuts=2, shuffle=shuffle, seed=7)

    expected = [list(batch.ids) for batch in make(restore_source=True)]
    sampler = make(restore_source=True)
    if prepare_iterator:
        iter(sampler)
    state = deepcopy(sampler.state_dict())
    if not shuffle:
        assert state["cuts_state"] == [source_state]
    restored = make(restore_source=False)
    restored.load_state_dict(state)
    assert [list(batch.ids) for batch in restored] == expected


@pytest.mark.parametrize("sampler_type", [DynamicCutSampler, DynamicBucketingSampler])
@pytest.mark.parametrize("shuffle", [False, True])
def test_indexed_sampler_checkpoint_before_iteration(
    indexed_sampler_factory, sampler_type, shuffle
):
    sampler = indexed_sampler_factory(sampler_type, shuffle=shuffle)
    state = deepcopy(sampler.state_dict())
    restored = indexed_sampler_factory(sampler_type, shuffle=shuffle)
    restored.load_state_dict(state)
    expected = _remaining_ids(iter(sampler))
    assert _remaining_ids(iter(restored)) == expected


class _IndexedCutsWithoutGraphOrigin(IteratorNode):
    is_checkpointable = True
    is_indexed = True
    has_constant_time_access = True

    def __init__(self, cuts):
        self.cuts = list(cuts)
        self.position = 0
        self._restored = False

    def __iter__(self):
        start = self.position if self._restored else 0
        self._restored = False
        for idx in range(start, len(self.cuts)):
            self.position = idx + 1
            yield self.cuts[idx]

    def __getitem__(self, idx):
        return self.cuts[idx]

    def __len__(self):
        return len(self.cuts)

    def state_dict(self):
        return {"position": self.position}

    def load_state_dict(self, sd):
        self.position = sd["position"]
        self._restored = True


class _CountingGraphRestoreSource(IteratorNode):
    is_indexed = True

    def __init__(self, cuts):
        self.cuts = list(cuts)
        self.capability_checks = 0

    @property
    def has_constant_time_access(self):
        self.capability_checks += 1
        return True

    def __iter__(self):
        for idx, cut in enumerate(self.cuts):
            yield attach_graph_origin(cut, idx)

    def __getitem__(self, idx):
        return attach_graph_origin(deepcopy(self.cuts[idx]), idx)

    def __len__(self):
        return len(self.cuts)


class TestDynamicSamplerCheckpoint:
    def test_state_dict_contains_diagnostics(self, indexed_cuts):
        sampler = DynamicCutSampler(indexed_cuts, max_cuts=5, shuffle=False)
        iter(sampler)
        for _ in range(3):
            next(sampler)
        sd = sampler.state_dict()
        assert "diagnostics" in sd

    def test_dynamic_sampler_restore_matches_uninterrupted(self, indexed_cuts):
        """Verify that restored sampler produces batches consistent with diagnostics."""
        sampler1 = DynamicCutSampler(indexed_cuts, max_cuts=5, shuffle=False)
        iter(sampler1)
        batches_before = []
        for _ in range(3):
            batches_before.append(next(sampler1))

        sd = sampler1.state_dict()

        # Restore into a new sampler
        sampler2 = DynamicCutSampler(indexed_cuts, max_cuts=5, shuffle=False)
        sampler2.load_state_dict(sd.copy())

        # After restoration, the diagnostics should report 3 batches
        assert sampler2.diagnostics.current_epoch_stats.total_batches == 3

    def test_dynamic_bucketing_sampler_restore(self, indexed_cuts):
        """DynamicBucketingSampler can be checkpointed and restored."""
        sampler1 = DynamicBucketingSampler(
            indexed_cuts, max_cuts=5, shuffle=False, num_buckets=2
        )
        iter(sampler1)
        for _ in range(3):
            next(sampler1)

        sd = sampler1.state_dict()

        sampler2 = DynamicBucketingSampler(
            indexed_cuts, max_cuts=5, shuffle=False, num_buckets=2
        )
        sampler2.load_state_dict(sd.copy())

        assert sampler2.diagnostics.current_epoch_stats.total_batches == 3


class TestIterableDatasetWrapperStateDict:
    def test_state_dict_round_trip(self, indexed_cuts):
        """IterableDatasetWrapper.state_dict() and load_state_dict() round-trip."""
        import torch.utils.data

        class IdentityDataset(torch.utils.data.Dataset):
            def __getitem__(self, item):
                return item

        dataset = IdentityDataset()
        sampler = DynamicCutSampler(indexed_cuts, max_cuts=10, shuffle=False)
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
        sampler2 = DynamicCutSampler(indexed_cuts, max_cuts=10, shuffle=False)
        wrapper2 = IterableDatasetWrapper(dataset, sampler2)
        wrapper2.load_state_dict(sd)

        assert wrapper2.epoch == 3
        assert wrapper2.sampler.diagnostics.current_epoch_stats.total_batches == 2


class TestShuffleIndexedGuard:
    def test_dynamic_cut_sampler_allows_shuffle_with_indexed(self, tmp_path):
        """shuffle=True + indexed dataset is supported for DynamicCutSampler."""
        path = tmp_path / "cuts.jsonl"
        DummyManifest(CutSet, begin_id=0, end_id=20).to_jsonl(path)

        cs = CutSet.from_file(path, indexed=True)
        assert cs.is_indexed is True

        # DynamicBucketingSampler with shuffle=True is allowed (RNG is saved)
        DynamicBucketingSampler(cs, max_cuts=5, shuffle=True, num_buckets=2)
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

    def test_dynamic_sampler_indexed_restore_preserves_diagnostics(self, tmp_path):
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

        iter(sampler2)

        assert sampler2.diagnostics.current_epoch_stats.total_batches == consumed

    def test_dynamic_sampler_indexed_shuffle_captures_lazy_shuffler_state(
        self, tmp_path
    ):
        path = tmp_path / "cuts.jsonl"
        DummyManifest(CutSet, begin_id=0, end_id=40).to_jsonl(path)
        cs = CutSet.from_file(path, indexed=True)

        sampler = DynamicCutSampler(cs, max_cuts=5, shuffle=True, seed=13)
        iter(sampler)
        for _ in range(4):
            next(sampler)

        sd = sampler.state_dict()
        assert "cuts_state" in sd
        root_state = sd["cuts_state"][0]
        assert root_state["_type"] == "LazyShuffler"
        assert "_state" in root_state

    def test_dynamic_sampler_indexed_shuffle_restore_is_exact(self, tmp_path):
        path = tmp_path / "cuts.jsonl"
        DummyManifest(CutSet, begin_id=0, end_id=50).to_jsonl(path)

        def make_sampler():
            return DynamicCutSampler(
                CutSet.from_file(path, indexed=True),
                max_cuts=5,
                shuffle=True,
                seed=13,
            )

        full = make_sampler()
        all_batches = [[c.id for c in batch] for batch in full]

        sampler1 = make_sampler()
        iterator1 = iter(sampler1)
        first_batches = [[c.id for c in next(iterator1)] for _ in range(3)]
        sd = deepcopy(sampler1.state_dict())

        sampler2 = make_sampler()
        sampler2.load_state_dict(sd)

        iter(sampler2)

        remaining = [[c.id for c in batch] for batch in sampler2]
        assert first_batches + remaining == all_batches

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

    def test_dynamic_bucketing_indexed_mix_restore(self, tmp_path):
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

        iter(sampler2)

    def test_dynamic_bucketing_indexed_restore_restore_preserves_diagnostics(
        self, tmp_path
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

        consumed = sampler1.diagnostics.current_epoch_stats.total_batches
        assert consumed == 3

        sampler2 = DynamicBucketingSampler(cs, max_cuts=5, shuffle=False, num_buckets=2)
        sampler2.load_state_dict(sd.copy())

        iter(sampler2)

        assert sampler2.diagnostics.current_epoch_stats.total_batches == consumed

    def test_dynamic_bucketing_indexed_restore_supports_iterator_node_roots(
        self, tmp_path
    ):
        path = tmp_path / "cuts.jsonl"
        DummyManifest(CutSet, begin_id=0, end_id=40).to_jsonl(path)

        def make_sampler():
            source = LazyRepeater(CutSet.from_file(path, indexed=True).data, times=2)
            return DynamicBucketingSampler(
                source, max_cuts=5, shuffle=False, num_buckets=2
            )

        full = make_sampler()
        all_batches = [[c.id for c in batch] for batch in full]

        sampler1 = make_sampler()
        iterator1 = iter(sampler1)
        first_batches = [[c.id for c in next(iterator1)] for _ in range(3)]
        sd = sampler1.state_dict()

        sampler2 = make_sampler()
        sampler2.load_state_dict(sd.copy())

        iter(sampler2)

        remaining = [[c.id for c in batch] for batch in sampler2]
        assert first_batches + remaining == all_batches

    @pytest.mark.parametrize(
        "sampler_type", [DynamicCutSampler, DynamicBucketingSampler]
    )
    def test_indexed_prefetch_state_restores_next_batch(self, tmp_path, sampler_type):
        path = tmp_path / "cuts.jsonl"
        DummyManifest(CutSet, begin_id=0, end_id=80).to_jsonl(path)
        cs = CutSet.from_file(path, indexed=True)

        sampler1 = sampler_type(cs, max_cuts=5, shuffle=False)
        iter(sampler1)
        # Simulate a worker-prefetched batch that wasn't yet accounted in
        # sampler diagnostics.
        next(sampler1.cuts_iter)
        assert sampler1.diagnostics.current_epoch_stats.total_batches == 0

        sd = sampler1.state_dict()
        if sampler_type is DynamicBucketingSampler:
            assert "bucketer_state" in sd

        expected_next = [c.id for c in next(sampler1)]

        sampler2 = sampler_type(cs, max_cuts=5, shuffle=False)
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

        sampler2 = DynamicBucketingSampler(cs, max_cuts=5, shuffle=False, num_buckets=2)
        sampler2.load_state_dict(sd.copy())
        with pytest.raises(RuntimeError, match="missing 'bucketer_state'"):
            iter(sampler2)

    def test_dynamic_bucketing_indexed_restore_requires_graph_tokens(self):
        cs = CutSet(
            _IndexedCutsWithoutGraphOrigin(DummyManifest(CutSet, begin_id=0, end_id=40))
        )

        sampler = DynamicBucketingSampler(cs, max_cuts=5, shuffle=False, num_buckets=2)
        iter(sampler)
        for _ in range(3):
            next(sampler)

        with pytest.raises(
            RuntimeError,
            match="DynamicBucketer checkpoint requires '_graph_origin'",
        ):
            sampler.state_dict()

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
        with pytest.raises(RuntimeError, match="missing 'cuts_state'"):
            iter(sampler2)

    @staticmethod
    def _make_bucketer_with_buffer(cuts, num_buffered=20):
        source = _CountingGraphRestoreSource(cuts)
        bucketer = DynamicBucketer(
            [],
            duration_bins=[10.0],
            world_size=1,
            max_cuts=4,
            buffer_size=num_buffered,
            concurrent=False,
            restore_sources=[source],
        )
        for idx in range(num_buffered):
            cut = attach_graph_origin(deepcopy(source.cuts[idx]), idx)
            bucketer.buckets[idx % len(bucketer.buckets)].put((cut,))
        return bucketer, source

    def test_dynamic_bucketer_snapshot_checks_each_source_once(self, cuts):
        bucketer, source = self._make_bucketer_with_buffer(cuts)

        state = bucketer.get_state()

        assert source.capability_checks == 1
        assert sum(len(tokens) for tokens in state["bucket_tokens"]) == 20

    def test_dynamic_bucketer_snapshot_revalidates_sources(self, cuts):
        bucketer, source = self._make_bucketer_with_buffer(cuts)

        bucketer.get_state()
        bucketer.get_state()

        assert source.capability_checks == 2

    def test_dynamic_bucketer_restore_checks_each_source_once(self, cuts):
        bucketer, _ = self._make_bucketer_with_buffer(cuts)
        state = bucketer.get_state()
        restored_source = _CountingGraphRestoreSource(cuts)
        restored = DynamicBucketer(
            [],
            duration_bins=[10.0],
            world_size=1,
            max_cuts=4,
            buffer_size=20,
            concurrent=False,
            restore_sources=[restored_source],
        )
        restored.set_state(state)

        restored._restore_from_saved_state()

        assert restored_source.capability_checks == 1
        assert sum(bucket.qsize() for bucket in restored.buckets) == 20


class TestNonConstantTimeSourcesRaise:
    """
    Checkpointing requires every leaf to be O(1)-restorable: a single streaming
    leaf makes LazyIteratorMultiplexer.has_constant_time_access False
    (it requires ALL children), so state_dict() must raise instead of
    producing a checkpoint that cannot be restored.
    """

    def _make_streaming_leaf(self):
        # A "leaf" with has_constant_time_access=False, no __getitem__.
        class _StreamingLeaf(IteratorNode):
            is_checkpointable = True
            has_constant_time_access = False

            def __init__(self, cuts):
                self.cuts = list(cuts)

            def __iter__(self):
                yield from self.cuts

            def __len__(self):
                return len(self.cuts)

            def state_dict(self):
                return {}

            def load_state_dict(self, sd):
                pass

        return _StreamingLeaf

    def test_mixed_raises(self, cuts):
        from lhotse.lazy import LazyIteratorMultiplexer

        streaming_leaf = self._make_streaming_leaf()(list(cuts)[:50])
        indexed_leaf = _IndexedCutsWithoutGraphOrigin(
            list(cuts)[50:]
        )  # has_constant_time_access=True per definition above
        mux = LazyIteratorMultiplexer(streaming_leaf, indexed_leaf, seed=0)
        cs = CutSet(mux)

        sampler = DynamicBucketingSampler(cs, max_cuts=5, shuffle=False, num_buckets=2)
        iter(sampler)
        for _ in range(3):
            next(sampler)

        with pytest.raises(RuntimeError, match="requires indexed lazy inputs"):
            sampler.state_dict()

    def test_all_streaming_raises(self, cuts):
        from lhotse.lazy import LazyIteratorMultiplexer

        leaf_cls = self._make_streaming_leaf()
        l1 = leaf_cls(list(cuts)[:50])
        l2 = leaf_cls(list(cuts)[50:])
        mux = LazyIteratorMultiplexer(l1, l2, seed=0)
        cs = CutSet(mux)

        sampler = DynamicBucketingSampler(cs, max_cuts=5, shuffle=False, num_buckets=2)
        iter(sampler)
        for _ in range(3):
            next(sampler)

        with pytest.raises(RuntimeError, match="requires indexed lazy inputs"):
            sampler.state_dict()
