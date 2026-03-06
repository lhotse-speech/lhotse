"""
Tests for the stateful iterator protocol (state_dict / load_state_dict)
added to lazy iterators in lhotse/lazy.py.

Every class gets two kinds of tests:
1. isinstance / attribute-name sanity checks (kept from original).
2. **Restoration correctness**: items yielded after restore == remaining
   items from an uninterrupted run.
"""
import json
import random
from contextlib import contextmanager
from pathlib import Path

import pytest

from lhotse.lazy import (
    IteratorNode,
    LazyFilter,
    LazyFlattener,
    LazyIteratorChain,
    LazyIteratorMultiplexer,
    LazyJsonlIterator,
    LazyManifestIterator,
    LazyMapper,
    LazyRepeater,
    LazyShuffler,
    LazySlicer,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextmanager
def jsonl_file(tmp_path, n=20, name="data.jsonl"):
    """Create a temp JSONL file with n records and auto-create the binary index."""
    from lhotse.indexing import create_jsonl_index

    p = tmp_path / name
    with open(p, "w") as f:
        for i in range(n):
            f.write(json.dumps({"id": f"item-{i}", "value": i}) + "\n")
    create_jsonl_index(p)
    yield p


def consume(it, k):
    """Consume k items from an iterator, return them as a list."""
    return [next(it) for _ in range(k)]


class _IndexedPlainIterator(IteratorNode):
    """
    Minimal indexed iterator used to test graph-token fallback paths.

    It supports ``__getitem__`` and checkpointing, but intentionally does not
    attach ``_graph_origin`` to yielded items.
    """

    is_checkpointable = True
    is_indexed = True
    has_constant_time_access = True

    def __init__(self, items):
        self.items = list(items)
        self.position = 0
        self._restored = False

    def __iter__(self):
        start = self.position if self._restored else 0
        self._restored = False
        for idx in range(start, len(self.items)):
            self.position = idx + 1
            yield self.items[idx]

    def __getitem__(self, idx):
        return self.items[idx]

    def __len__(self):
        return len(self.items)

    def state_dict(self):
        return {"position": self.position}

    def load_state_dict(self, sd):
        self.position = sd["position"]
        self._restored = True


# ---------------------------------------------------------------------------
# LazyJsonlIterator
# ---------------------------------------------------------------------------


class TestLazyJsonlIteratorStateful:
    def test_isinstance_stateful(self, tmp_path):
        with jsonl_file(tmp_path, n=5) as p:
            it = LazyJsonlIterator(p)
            assert isinstance(it, IteratorNode)
            assert not it.is_checkpointable

    def test_restore_yields_remaining_items(self, tmp_path):
        with jsonl_file(tmp_path, n=10) as p:
            # Full uninterrupted run
            all_items = list(LazyJsonlIterator(p))

            # Interrupted: consume 5, checkpoint
            it1 = LazyJsonlIterator(p)
            gen1 = iter(it1)
            first_k = consume(gen1, 5)
            sd = it1.state_dict()

            # Restore and iterate remaining
            it2 = LazyJsonlIterator(p)
            it2.load_state_dict(sd)
            remaining = list(it2)

            assert first_k + remaining == all_items

    def test_restore_at_start(self, tmp_path):
        """Restoring at position 0 yields all items."""
        with jsonl_file(tmp_path, n=10) as p:
            all_items = list(LazyJsonlIterator(p))

            it = LazyJsonlIterator(p)
            it.load_state_dict({"position": 0})
            assert list(it) == all_items

    def test_restore_at_end(self, tmp_path):
        """Restoring past the last item yields nothing."""
        with jsonl_file(tmp_path, n=10) as p:
            it = LazyJsonlIterator(p)
            it.load_state_dict({"position": 10})
            assert list(it) == []

    def test_restore_without_index(self, tmp_path):
        """Same restoration test but without the .idx file (sequential skip)."""
        p = tmp_path / "noidx.jsonl"
        with open(p, "w") as f:
            for i in range(10):
                f.write(json.dumps({"id": f"item-{i}", "value": i}) + "\n")
        # No create_jsonl_index call — forces the non-indexed path

        all_items = list(LazyJsonlIterator(p))

        it1 = LazyJsonlIterator(p)
        gen1 = iter(it1)
        first_k = consume(gen1, 5)
        sd = it1.state_dict()

        it2 = LazyJsonlIterator(p)
        it2.load_state_dict(sd)
        remaining = list(it2)

        assert first_k + remaining == all_items


# ---------------------------------------------------------------------------
# LazyManifestIterator
# ---------------------------------------------------------------------------


class TestLazyManifestIteratorStateful:
    def test_isinstance_stateful(self, tmp_path):
        with jsonl_file(tmp_path, n=5) as p:
            it = LazyManifestIterator(p)
            assert isinstance(it, IteratorNode)
            assert it.is_checkpointable

    def test_state_dict_delegates(self, tmp_path):
        """LazyManifestIterator delegates state_dict to its source."""
        with jsonl_file(tmp_path, n=10) as p:
            it = LazyManifestIterator(p)
            sd = it.state_dict()
            assert "source" in sd
            assert sd["source"] == {"position": 0}

    def test_restore_yields_remaining_items(self, tmp_path):
        """Use supervision-segment JSONL so deserialize_item works."""
        p = tmp_path / "supervisions.jsonl"
        with open(p, "w") as f:
            for i in range(10):
                seg = {
                    "id": f"seg-{i}",
                    "recording_id": "rec-0",
                    "start": float(i),
                    "duration": 1.0,
                }
                f.write(json.dumps(seg) + "\n")
        from lhotse.indexing import create_jsonl_index

        create_jsonl_index(p)

        all_items = list(LazyManifestIterator(p))

        it1 = LazyManifestIterator(p)
        gen1 = iter(it1)
        first_k = consume(gen1, 5)
        sd = it1.state_dict()

        it2 = LazyManifestIterator(p)
        it2.load_state_dict(sd)
        remaining = list(it2)

        assert first_k + remaining == all_items


# ---------------------------------------------------------------------------
# LazyIteratorChain
# ---------------------------------------------------------------------------


class TestLazyIteratorChainStateful:
    def test_isinstance_stateful(self):
        chain = LazyIteratorChain([1, 2], [3, 4])
        assert isinstance(chain, IteratorNode)
        assert chain.is_checkpointable

    def test_sources_attribute(self):
        chain = LazyIteratorChain([1, 2], [3, 4])
        assert hasattr(chain, "sources")
        assert len(chain.sources) == 2

    def test_auto_flatten(self):
        inner = LazyIteratorChain([1, 2], [3, 4])
        outer = LazyIteratorChain(inner, [5, 6])
        assert len(outer.sources) == 3
        assert list(outer) == [1, 2, 3, 4, 5, 6]

    def test_restore_yields_remaining_items(self, tmp_path):
        with jsonl_file(tmp_path, n=5, name="a.jsonl") as p1, jsonl_file(
            tmp_path, n=5, name="b.jsonl"
        ) as p2:
            s1a = LazyJsonlIterator(p1)
            s2a = LazyJsonlIterator(p2)
            chain_full = LazyIteratorChain(s1a, s2a)
            all_items = list(chain_full)
            assert len(all_items) == 10

            # Interrupted: consume 7 items (spans into second shard)
            s1b = LazyJsonlIterator(p1)
            s2b = LazyJsonlIterator(p2)
            chain1 = LazyIteratorChain(s1b, s2b)
            gen1 = iter(chain1)
            first_k = consume(gen1, 7)
            sd = chain1.state_dict()

            # Restore
            s1c = LazyJsonlIterator(p1)
            s2c = LazyJsonlIterator(p2)
            chain2 = LazyIteratorChain(s1c, s2c)
            chain2.load_state_dict(sd)
            remaining = list(chain2)

            assert first_k + remaining == all_items

    def test_restore_within_first_shard(self, tmp_path):
        with jsonl_file(tmp_path, n=5, name="a.jsonl") as p1, jsonl_file(
            tmp_path, n=5, name="b.jsonl"
        ) as p2:
            s1a = LazyJsonlIterator(p1)
            s2a = LazyJsonlIterator(p2)
            chain_full = LazyIteratorChain(s1a, s2a)
            all_items = list(chain_full)

            # Consume only 3 items (still in first shard)
            s1b = LazyJsonlIterator(p1)
            s2b = LazyJsonlIterator(p2)
            chain1 = LazyIteratorChain(s1b, s2b)
            gen1 = iter(chain1)
            first_k = consume(gen1, 3)
            sd = chain1.state_dict()

            s1c = LazyJsonlIterator(p1)
            s2c = LazyJsonlIterator(p2)
            chain2 = LazyIteratorChain(s1c, s2c)
            chain2.load_state_dict(sd)
            remaining = list(chain2)

            assert first_k + remaining == all_items

    def test_restore_with_shuffled_shards(self, tmp_path):
        """When shuffle_iters=True and seed is set, shard order is randomized.
        Restoration must replay the exact saved shard order."""
        with jsonl_file(tmp_path, n=4, name="a.jsonl") as p1, jsonl_file(
            tmp_path, n=4, name="b.jsonl"
        ) as p2, jsonl_file(tmp_path, n=4, name="c.jsonl") as p3:
            # Full run with shuffled shards
            chain_full = LazyIteratorChain(
                LazyJsonlIterator(p1),
                LazyJsonlIterator(p2),
                LazyJsonlIterator(p3),
                shuffle_iters=True,
                seed=42,
            )
            all_items = list(chain_full)
            assert len(all_items) == 12

            # Interrupted: consume 7 items (spans at least 2 shuffled shards)
            chain1 = LazyIteratorChain(
                LazyJsonlIterator(p1),
                LazyJsonlIterator(p2),
                LazyJsonlIterator(p3),
                shuffle_iters=True,
                seed=42,
            )
            gen1 = iter(chain1)
            first_k = consume(gen1, 7)
            sd = chain1.state_dict()

            # Restore
            chain2 = LazyIteratorChain(
                LazyJsonlIterator(p1),
                LazyJsonlIterator(p2),
                LazyJsonlIterator(p3),
                shuffle_iters=True,
                seed=42,
            )
            chain2.load_state_dict(sd)
            remaining = list(chain2)

            assert first_k + remaining == all_items

    def test_restore_with_duplicate_equality_sources(self):
        """Regression: iter_order must track source indices, not list.index(...)."""

        class EqualStateful(IteratorNode):
            is_checkpointable = True

            def __init__(self, values):
                self._values = list(values)
                self._position = 0
                self._restored = False

            def __iter__(self):
                start = self._position if self._restored else 0
                self._restored = False
                self._position = start
                for i in range(start, len(self._values)):
                    self._position = i + 1
                    yield self._values[i]

            def __len__(self):
                return len(self._values)

            def state_dict(self):
                return {"position": self._position}

            def load_state_dict(self, sd):
                self._position = sd["position"]
                self._restored = True

            def __eq__(self, other):
                return isinstance(other, EqualStateful)

        def make():
            return LazyIteratorChain(
                EqualStateful([1, 2, 3]), EqualStateful([10, 11, 12])
            )

        all_items = list(make())
        assert all_items == [1, 2, 3, 10, 11, 12]

        chain1 = make()
        gen1 = iter(chain1)
        first_k = consume(gen1, 4)
        sd = chain1.state_dict()

        chain2 = make()
        chain2.load_state_dict(sd)
        remaining = list(chain2)
        assert first_k + remaining == all_items

    def test_global_shuffle_accepts_randomized_seed_for_indexed(self, tmp_path):
        """Regression: globally shuffled indexed chain should handle non-int seeds."""
        from lhotse import CutSet
        from lhotse.lazy import LazyIndexedManifestIterator
        from lhotse.testing.dummies import DummyManifest

        p1 = tmp_path / "a.jsonl"
        p2 = tmp_path / "b.jsonl"
        DummyManifest(CutSet, begin_id=0, end_id=5).to_jsonl(p1)
        DummyManifest(CutSet, begin_id=100, end_id=105).to_jsonl(p2)

        chain = LazyIteratorChain(
            LazyIndexedManifestIterator(p1),
            LazyIndexedManifestIterator(p2),
            shuffle_iters=True,
            seed="randomized",
        )
        items = list(chain)
        assert len(items) == 10

    def test_restore_does_not_poison_next_epoch(self, tmp_path):
        """Regression: load_state_dict must only restore sources that will
        be iterated (at/after current_iter_idx).  Previously, ALL sources
        were restored, leaving skipped sources with a stale _restored flag
        that caused them to start from the saved position instead of fresh
        on the next epoch."""
        with jsonl_file(tmp_path, n=5, name="a.jsonl") as p1, jsonl_file(
            tmp_path, n=5, name="b.jsonl"
        ) as p2:

            def make():
                s1 = LazyJsonlIterator(p1)
                s2 = LazyJsonlIterator(p2)
                chain = LazyIteratorChain(s1, s2)
                rep = LazyRepeater(chain, times=2)
                return rep

            # Full run: 10 items × 2 epochs = 20
            all_items = list(make())
            assert len(all_items) == 20

            # Interrupted: consume 7 items (into shard B of epoch 0)
            pipe1 = make()
            gen1 = iter(pipe1)
            first_k = consume(gen1, 7)
            sd = pipe1.state_dict()

            # Restore
            pipe2 = make()
            pipe2.load_state_dict(sd)
            remaining = list(pipe2)

            assert first_k + remaining == all_items


# ---------------------------------------------------------------------------
# LazyIteratorMultiplexer
# ---------------------------------------------------------------------------


class TestLazyIteratorMultiplexerStateful:
    def test_isinstance_stateful(self):
        mux = LazyIteratorMultiplexer([1, 2], [3, 4])
        assert isinstance(mux, IteratorNode)
        assert mux.is_checkpointable

    def test_sources_attribute(self):
        mux = LazyIteratorMultiplexer([1, 2, 3], [4, 5, 6])
        assert hasattr(mux, "sources")
        assert len(mux.sources) == 2

    def test_restore_yields_remaining_items(self, tmp_path):
        with jsonl_file(tmp_path, n=8, name="a.jsonl") as p1, jsonl_file(
            tmp_path, n=8, name="b.jsonl"
        ) as p2:
            s1a = LazyJsonlIterator(p1)
            s2a = LazyJsonlIterator(p2)
            mux_full = LazyIteratorMultiplexer(s1a, s2a, seed=42)
            all_items = list(mux_full)

            # Interrupted: consume 5
            s1b = LazyJsonlIterator(p1)
            s2b = LazyJsonlIterator(p2)
            mux1 = LazyIteratorMultiplexer(s1b, s2b, seed=42)
            gen1 = iter(mux1)
            first_k = consume(gen1, 5)
            sd = mux1.state_dict()

            # Restore
            s1c = LazyJsonlIterator(p1)
            s2c = LazyJsonlIterator(p2)
            mux2 = LazyIteratorMultiplexer(s1c, s2c, seed=42)
            mux2.load_state_dict(sd)
            remaining = list(mux2)

            assert first_k + remaining == all_items

    def test_restore_without_child_graph_tokens(self):
        mux_full = LazyIteratorMultiplexer(
            _IndexedPlainIterator(range(8)),
            _IndexedPlainIterator(range(100, 108)),
            seed=42,
        )
        all_items = list(mux_full)

        mux1 = LazyIteratorMultiplexer(
            _IndexedPlainIterator(range(8)),
            _IndexedPlainIterator(range(100, 108)),
            seed=42,
        )
        gen1 = iter(mux1)
        first_k = consume(gen1, 5)
        sd = mux1.state_dict()

        mux2 = LazyIteratorMultiplexer(
            _IndexedPlainIterator(range(8)),
            _IndexedPlainIterator(range(100, 108)),
            seed=42,
        )
        mux2.load_state_dict(sd)
        remaining = list(mux2)

        assert first_k + remaining == all_items


# ---------------------------------------------------------------------------
# LazyInfiniteApproximateMultiplexer
# ---------------------------------------------------------------------------


class TestLazyInfiniteApproximateMultiplexerStateful:
    def test_not_stateful_iterator(self):
        from lhotse.lazy import LazyInfiniteApproximateMultiplexer

        mux = LazyInfiniteApproximateMultiplexer([1, 2], [3, 4])
        assert isinstance(mux, IteratorNode)
        assert not mux.is_checkpointable

    def test_sources_attribute(self):
        from lhotse.lazy import LazyInfiniteApproximateMultiplexer

        mux = LazyInfiniteApproximateMultiplexer([1, 2, 3], [4, 5, 6])
        assert hasattr(mux, "sources")
        assert len(mux.sources) == 2

    def test_collect_state_dict_raises(self):
        from lhotse.checkpoint import collect_state_dict
        from lhotse.lazy import LazyInfiniteApproximateMultiplexer

        mux = LazyInfiniteApproximateMultiplexer([1, 2], [3, 4])
        with pytest.raises(NotImplementedError, match="does not support checkpointing"):
            collect_state_dict(mux)


# ---------------------------------------------------------------------------
# LazyShuffler
# ---------------------------------------------------------------------------


class TestLazyShufflerStateful:
    def test_not_stateful_iterator(self):
        shuf = LazyShuffler([1, 2, 3])
        assert isinstance(shuf, IteratorNode)
        assert not shuf.is_checkpointable

    def test_source_attribute(self):
        shuf = LazyShuffler([1, 2, 3])
        assert hasattr(shuf, "source")

    def test_collect_state_dict_raises(self):
        from lhotse.checkpoint import collect_state_dict

        shuf = LazyShuffler([1, 2, 3])
        with pytest.raises(NotImplementedError, match="does not support checkpointing"):
            collect_state_dict(shuf)


# ---------------------------------------------------------------------------
# LazyFilter
# ---------------------------------------------------------------------------


class TestLazyFilterStateful:
    def test_isinstance_stateful(self):
        filt = LazyFilter([1, 2, 3], predicate=lambda x: True)
        assert isinstance(filt, IteratorNode)
        assert filt.is_checkpointable

    def test_source_attribute(self):
        filt = LazyFilter([1, 2, 3], predicate=lambda x: x > 1)
        assert hasattr(filt, "source")

    def test_restore_yields_remaining_items(self, tmp_path):
        predicate = lambda x: x["value"] % 2 == 0

        with jsonl_file(tmp_path, n=10) as p:
            all_items = list(LazyFilter(LazyJsonlIterator(p), predicate=predicate))

            inner1 = LazyJsonlIterator(p)
            filt1 = LazyFilter(inner1, predicate=predicate)
            gen1 = iter(filt1)
            first_k = consume(gen1, 3)
            sd = filt1.state_dict()

            inner2 = LazyJsonlIterator(p)
            filt2 = LazyFilter(inner2, predicate=predicate)
            filt2.load_state_dict(sd)
            remaining = list(filt2)

            assert first_k + remaining == all_items


# ---------------------------------------------------------------------------
# LazyMapper
# ---------------------------------------------------------------------------


class TestLazyMapperStateful:
    def test_isinstance_stateful(self):
        mapper = LazyMapper([1, 2, 3], fn=lambda x: x)
        assert isinstance(mapper, IteratorNode)
        assert mapper.is_checkpointable

    def test_source_attribute(self):
        mapper = LazyMapper([1, 2, 3], fn=lambda x: x * 2)
        assert hasattr(mapper, "source")

    def test_restore_yields_remaining_items(self, tmp_path):
        fn = lambda x: {**x, "value": x["value"] * 10}

        with jsonl_file(tmp_path, n=10) as p:
            all_items = list(LazyMapper(LazyJsonlIterator(p), fn=fn))

            inner1 = LazyJsonlIterator(p)
            mapper1 = LazyMapper(inner1, fn=fn)
            gen1 = iter(mapper1)
            first_k = consume(gen1, 4)
            sd = mapper1.state_dict()

            inner2 = LazyJsonlIterator(p)
            mapper2 = LazyMapper(inner2, fn=fn)
            mapper2.load_state_dict(sd)
            remaining = list(mapper2)

            assert first_k + remaining == all_items


# ---------------------------------------------------------------------------
# LazyFlattener
# ---------------------------------------------------------------------------


class TestLazyFlattenerStateful:
    def test_checkpointing_not_supported(self):
        flat = LazyFlattener([[1, 2], [3, 4]])
        with pytest.raises(NotImplementedError, match="does not support checkpointing"):
            flat.state_dict()
        with pytest.raises(NotImplementedError, match="does not support checkpointing"):
            flat.load_state_dict({})

    def test_source_attribute(self):
        flat = LazyFlattener([[1, 2], [3, 4]])
        assert hasattr(flat, "source")

    def test_collect_state_dict_raises_for_flattener(self):
        from lhotse.checkpoint import collect_state_dict

        flat = LazyFlattener([[1, 2], [3, 4]])
        with pytest.raises(NotImplementedError, match="does not support checkpointing"):
            collect_state_dict(flat)


# ---------------------------------------------------------------------------
# LazyRepeater
# ---------------------------------------------------------------------------


class TestLazyRepeaterStateful:
    def test_isinstance_stateful(self):
        rep = LazyRepeater([1, 2], times=2)
        assert isinstance(rep, IteratorNode)
        assert rep.is_checkpointable

    def test_source_attribute(self):
        rep = LazyRepeater([1, 2, 3], times=2)
        assert hasattr(rep, "source")

    def test_restore_yields_remaining_items(self, tmp_path):
        with jsonl_file(tmp_path, n=5) as p:
            # Full run: 3 epochs × 5 items = 15
            inner_full = LazyJsonlIterator(p)
            rep_full = LazyRepeater(inner_full, times=3, preserve_id=True)
            all_items = list(rep_full)
            assert len(all_items) == 15

            # Interrupted: consume 8 items (into epoch 1)
            inner1 = LazyJsonlIterator(p)
            rep1 = LazyRepeater(inner1, times=3, preserve_id=True)
            gen1 = iter(rep1)
            first_k = consume(gen1, 8)
            sd = rep1.state_dict()

            # Restore
            inner2 = LazyJsonlIterator(p)
            rep2 = LazyRepeater(inner2, times=3, preserve_id=True)
            rep2.load_state_dict(sd)
            remaining = list(rep2)

            assert first_k + remaining == all_items

    def test_epoch_tracking(self):
        rep = LazyRepeater([1, 2], times=3)
        gen = iter(rep)
        consume(gen, 2)
        assert rep._current_epoch == 0
        consume(gen, 2)
        assert rep._current_epoch == 1


# ---------------------------------------------------------------------------
# LazySlicer
# ---------------------------------------------------------------------------


class TestLazySlicerStateful:
    def test_isinstance_stateful(self):
        slicer = LazySlicer([1, 2, 3], k=0, n=2)
        assert isinstance(slicer, IteratorNode)
        assert slicer.is_checkpointable

    def test_source_attribute(self):
        slicer = LazySlicer([1, 2, 3, 4, 5, 6], k=0, n=2)
        assert hasattr(slicer, "source")

    def test_restore_yields_remaining_items(self, tmp_path):
        with jsonl_file(tmp_path, n=12) as p:
            # k=0, n=3 picks every 3rd item starting at index 0 → 4 items
            all_items = list(LazySlicer(LazyJsonlIterator(p), k=0, n=3))
            assert len(all_items) == 4

            inner1 = LazyJsonlIterator(p)
            slicer1 = LazySlicer(inner1, k=0, n=3)
            gen1 = iter(slicer1)
            first_k = consume(gen1, 2)
            sd = slicer1.state_dict()

            inner2 = LazyJsonlIterator(p)
            slicer2 = LazySlicer(inner2, k=0, n=3)
            slicer2.load_state_dict(sd)
            remaining = list(slicer2)

            assert first_k + remaining == all_items


# ---------------------------------------------------------------------------
# Nested graph test
# ---------------------------------------------------------------------------


class TestNestedStatefulGraph:
    def test_nested_state_dict(self, tmp_path):
        """
        Test state_dict on a nested pipeline:
        LazyRepeater(LazyMapper(LazyIteratorChain(...)))
        """
        with jsonl_file(tmp_path, n=5, name="a.jsonl") as p1, jsonl_file(
            tmp_path, n=5, name="b.jsonl"
        ) as p2:
            s1 = LazyJsonlIterator(p1)
            s2 = LazyJsonlIterator(p2)
            chain = LazyIteratorChain(s1, s2)
            mapper = LazyMapper(chain, fn=lambda x: x)
            rep = LazyRepeater(mapper, times=2)

            sd = rep.state_dict()
            assert "current_epoch" in sd
            assert "source" in sd  # LazyMapper state
            assert "source" in sd["source"]  # LazyIteratorChain state

    def test_nested_load_state_dict(self, tmp_path):
        """Round-trip state_dict -> load_state_dict on a nested pipeline."""
        with jsonl_file(tmp_path, n=5, name="a.jsonl") as p1, jsonl_file(
            tmp_path, n=5, name="b.jsonl"
        ) as p2:
            s1 = LazyJsonlIterator(p1)
            s2 = LazyJsonlIterator(p2)
            chain = LazyIteratorChain(s1, s2)
            mapper = LazyMapper(chain, fn=lambda x: x)
            rep = LazyRepeater(mapper, times=2)

            sd = rep.state_dict()

            # Build fresh pipeline
            s1b = LazyJsonlIterator(p1)
            s2b = LazyJsonlIterator(p2)
            chain2 = LazyIteratorChain(s1b, s2b)
            mapper2 = LazyMapper(chain2, fn=lambda x: x)
            rep2 = LazyRepeater(mapper2, times=2)

            # Should not raise
            rep2.load_state_dict(sd)

    def test_chain_of_jsonl_restore(self, tmp_path):
        """End-to-end: chain -> checkpoint -> restore yields correct remaining items."""
        with jsonl_file(tmp_path, n=6, name="a.jsonl") as p1, jsonl_file(
            tmp_path, n=6, name="b.jsonl"
        ) as p2:
            # Full run
            s1a = LazyJsonlIterator(p1)
            s2a = LazyJsonlIterator(p2)
            all_items = list(LazyIteratorChain(s1a, s2a))
            assert len(all_items) == 12

            # Interrupt mid-second-shard
            s1b = LazyJsonlIterator(p1)
            s2b = LazyJsonlIterator(p2)
            chain1 = LazyIteratorChain(s1b, s2b)
            gen1 = iter(chain1)
            first_k = consume(gen1, 9)
            sd = chain1.state_dict()

            # Restore
            s1c = LazyJsonlIterator(p1)
            s2c = LazyJsonlIterator(p2)
            chain2 = LazyIteratorChain(s1c, s2c)
            chain2.load_state_dict(sd)
            remaining = list(chain2)

            assert first_k + remaining == all_items
