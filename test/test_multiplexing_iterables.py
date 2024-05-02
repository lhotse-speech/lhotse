import pickle

from lhotse import CutSet
from lhotse.lazy import LazyInfiniteApproximateMultiplexer, LazyIteratorMultiplexer
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
        return f"dummy-mono-cut-{i:04d}"

    assert sorted([c.id for c in cuts_mux]) == [
        cid(i) for i in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1000, 1001, 1002, 1003, 1004)
    ]
    assert sorted([c.id for c in cuts_mux]) != [c.id for c in cuts_mux]


def test_cut_set_mux_stop_early():
    cuts1 = DummyManifest(CutSet, begin_id=0, end_id=10)
    cuts2 = DummyManifest(CutSet, begin_id=1000, end_id=1005)

    cuts_mux = CutSet.mux(cuts1, cuts2, seed=0, stop_early=True)

    def cid(i: int) -> str:
        return f"dummy-mono-cut-{i:04d}"

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


def test_multiplexer_max_open_streams():
    mux = LazyInfiniteApproximateMultiplexer(
        range(3),
        range(10, 13),
        range(100, 103),
        seed=1,
        max_open_streams=2,
    )

    it = iter(mux)
    samples = []
    for _ in range(9):
        samples.append(next(it))

    # Remember we are sampling with replacement when using
    # max_open_streams. Here, the following streams were picked:
    # stream2 [100-102],
    # stream0 [0-2]
    # stream0 [0-2]
    # stream1 [10-12]
    assert samples == [100, 0, 1, 2, 101, 102, 0, 10, 11]


def test_multiplexer_max_open_streams_1():
    mux = LazyInfiniteApproximateMultiplexer(
        range(3),
        range(10, 13),
        range(100, 103),
        seed=1,
        max_open_streams=1,
    )

    it = iter(mux)
    samples = []
    for _ in range(9):
        samples.append(next(it))

    # When max_open_streams=1, mux is reduced to a chain
    assert samples == [0, 1, 2, 10, 11, 12, 0, 1, 2]


def test_cut_set_infinite_mux():
    cuts1 = DummyManifest(CutSet, begin_id=0, end_id=3)
    cuts2 = DummyManifest(CutSet, begin_id=10, end_id=13)
    cuts3 = DummyManifest(CutSet, begin_id=100, end_id=103)

    cuts_mux = CutSet.infinite_mux(
        cuts1, cuts2, cuts3, seed=0, max_open_streams=2, weights=[0.6, 0.3, 0.1]
    )

    def cid(i: int) -> str:
        return f"dummy-mono-cut-{i:04d}"

    samples = []
    for i, cut in enumerate(cuts_mux):
        if i == 20:
            break
        samples.append(cut)

    expected_ids = (
        10,
        11,
        10,
        12,
        11,
        0,
        1,
        12,
        2,
        10,
        0,
        1,
        2,
        100,
        11,
        12,
        101,
        0,
        1,
        2,
    )

    assert [c.id for c in samples] == [cid(i) for i in expected_ids]
