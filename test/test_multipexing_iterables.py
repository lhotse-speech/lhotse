import pickle

from lhotse import CutSet
from lhotse.lazy import LazyIteratorMultiplexer
from lhotse.testing.dummies import DummyManifest


def test_multiplexer():
    mux = LazyIteratorMultiplexer(range(10), range(900, 903), seed=0)  # len 10  # len 3

    assert sorted(list(mux)) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 900, 901, 902]
    assert sorted(list(mux)) != list(mux)


def test_multiplexer_deterministic():
    # seed given
    mux = LazyIteratorMultiplexer(
        range(1000), range(900000, 901000), seed=0  # len 10  # len 3
    )
    assert list(mux) == list(mux)


def test_multiplexer_weights():
    mux_uniform = LazyIteratorMultiplexer(
        range(10), range(900, 903), seed=0  # len 10  # len 3
    )
    mux_weighted = LazyIteratorMultiplexer(
        range(10),  # len 10
        range(900, 903),  # len 3
        seed=0,
        weights=[10, 3],
    )

    assert sorted(list(mux_weighted)) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 900, 901, 902]
    assert sorted(list(mux_weighted)) != list(mux_weighted)
    assert list(mux_weighted) != list(mux_uniform)


def test_cut_set_mux():
    cuts1 = DummyManifest(CutSet, begin_id=0, end_id=10)
    cuts2 = DummyManifest(CutSet, begin_id=1000, end_id=1005)

    cuts_mux = CutSet.mux(cuts1, cuts2, seed=0)

    def cid(i: int) -> str:
        return f"dummy-cut-{i:04d}"

    assert sorted([c.id for c in cuts_mux]) == [
        cid(i) for i in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1000, 1001, 1002, 1003, 1004)
    ]
    assert sorted([c.id for c in cuts_mux]) != [c.id for c in cuts_mux]


def test_cut_set_mux_stop_early():
    cuts1 = DummyManifest(CutSet, begin_id=0, end_id=10)
    cuts2 = DummyManifest(CutSet, begin_id=1000, end_id=1005)

    cuts_mux = CutSet.mux(cuts1, cuts2, seed=0, stop_early=True)

    def cid(i: int) -> str:
        return f"dummy-cut-{i:04d}"

    assert sorted([c.id for c in cuts_mux]) == [
        cid(i) for i in (0, 1, 2, 3, 4, 1000, 1001, 1002, 1003, 1004)
    ]
    assert sorted([c.id for c in cuts_mux]) != [c.id for c in cuts_mux]


def test_multiplexer_pickling():
    mux = LazyIteratorMultiplexer(
        list(range(100)), list(range(10)), weights=[2, 3], seed=0
    )

    data = pickle.dumps(mux)
    mux_rec = pickle.loads(data)

    assert list(mux) == list(mux_rec)


def test_multiplexer_with_cuts_pickling():
    cuts1 = DummyManifest(CutSet, begin_id=0, end_id=10)
    cuts2 = DummyManifest(CutSet, begin_id=1000, end_id=1005)
    mux = LazyIteratorMultiplexer(cuts1, cuts2, seed=0)

    data = pickle.dumps(mux)
    mux_rec = pickle.loads(data)

    assert list(mux) == list(mux_rec)
