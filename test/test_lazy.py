"""
Since the tests in this module handle very different types of *Set classes,
we try to leverage 'duration' attribute which is shared by all tested types of items
(cuts, features, recordings, supervisions).
"""
import random
from concurrent.futures import ProcessPoolExecutor

import pytest

from lhotse import CutSet, FeatureSet, RecordingSet, SupervisionSet, combine
from lhotse.testing.dummies import DummyManifest, as_lazy
from lhotse.utils import fastcopy, is_module_available


@pytest.mark.parametrize(
    "manifest_type", [RecordingSet, SupervisionSet, FeatureSet, CutSet]
)
def test_combine_lazy(manifest_type):
    expected = DummyManifest(manifest_type, begin_id=0, end_id=200)
    with as_lazy(DummyManifest(manifest_type, begin_id=0, end_id=68)) as part1, as_lazy(
        DummyManifest(manifest_type, begin_id=68, end_id=136)
    ) as part2, as_lazy(
        DummyManifest(manifest_type, begin_id=136, end_id=200)
    ) as part3:
        combined = combine(part1, part2, part3)
        # Equivalent under iteration
        assert list(combined) == list(expected)


@pytest.mark.parametrize(
    "manifest_type", [RecordingSet, SupervisionSet, FeatureSet, CutSet]
)
def test_subset_first_lazy(manifest_type):
    any_set = DummyManifest(manifest_type, begin_id=0, end_id=200)
    expected = DummyManifest(manifest_type, begin_id=0, end_id=10)
    subset = any_set.subset(first=10)
    assert subset == expected


@pytest.mark.parametrize(
    "manifest_type", [RecordingSet, SupervisionSet, FeatureSet, CutSet]
)
def test_subset_last_lazy(manifest_type):
    any_set = DummyManifest(manifest_type, begin_id=0, end_id=200)
    expected = DummyManifest(manifest_type, begin_id=190, end_id=200)
    subset = any_set.subset(last=10)
    assert subset == expected


@pytest.mark.parametrize(
    "manifest_type", [RecordingSet, SupervisionSet, FeatureSet, CutSet]
)
@pytest.mark.parametrize(["first", "last"], [(None, None), (10, 10)])
def test_subset_raises_lazy(manifest_type, first, last):
    any_set = DummyManifest(manifest_type, begin_id=0, end_id=200)
    with pytest.raises(AssertionError):
        subset = any_set.subset(first=first, last=last)


@pytest.mark.parametrize(
    "manifest_type", [RecordingSet, SupervisionSet, FeatureSet, CutSet]
)
def test_map(manifest_type):

    expected = DummyManifest(manifest_type, begin_id=0, end_id=10)
    for item in expected:
        item.duration = 3.14

    def transform_fn(item):
        item.duration = 3.14
        return item

    data = DummyManifest(manifest_type, begin_id=0, end_id=10)
    eager_result = data.map(transform_fn)
    assert list(eager_result) == list(expected)

    with as_lazy(data) as lazy_data:
        lazy_result = lazy_data.map(transform_fn)
        assert list(lazy_result) == list(expected)


@pytest.mark.parametrize(
    "manifest_type", [RecordingSet, SupervisionSet, FeatureSet, CutSet]
)
def test_filter(manifest_type):

    expected = DummyManifest(manifest_type, begin_id=0, end_id=5)
    for idx, item in enumerate(expected):
        item.duration = idx

    def predicate(item):
        return item.duration < 5

    data = DummyManifest(manifest_type, begin_id=0, end_id=10)
    for idx, item in enumerate(data):
        item.duration = idx

    eager_result = data.filter(predicate)
    assert list(eager_result) == list(expected)

    with as_lazy(data) as lazy_data:
        lazy_result = lazy_data.filter(predicate)
        with pytest.raises(NotImplementedError):
            assert list(lazy_result) == list(expected)
        assert list(lazy_result.to_eager()) == list(expected)


@pytest.mark.parametrize(
    "manifest_type", [RecordingSet, SupervisionSet, FeatureSet, CutSet]
)
@pytest.mark.parametrize("preserve_id", [True, False])
def test_repeat(manifest_type, preserve_id):
    data = DummyManifest(manifest_type, begin_id=0, end_id=10)

    expected = data + data

    eager_result = data.repeat(times=2, preserve_id=preserve_id)
    if preserve_id or manifest_type == FeatureSet:
        assert list(eager_result) == list(expected)
    else:
        items = list(eager_result)
        ref_items = list(expected)
        assert len(items) == len(ref_items)
        for i, refi in zip(items, ref_items):
            assert i.id.endswith("_repeat0") or i.id.endswith("_repeat1")
            i_modi = fastcopy(i, id=refi.id)
            assert i_modi == refi

    with as_lazy(data) as lazy_data:
        lazy_result = lazy_data.repeat(times=2, preserve_id=preserve_id)
        if preserve_id or manifest_type == FeatureSet:
            assert list(lazy_result) == list(expected)
        else:
            items = list(lazy_result)
            ref_items = list(expected)
            assert len(items) == len(ref_items)
            for i, refi in zip(items, ref_items):
                assert i.id.endswith("_repeat0") or i.id.endswith("_repeat1")
                i_modi = fastcopy(i, id=refi.id)
                assert i_modi == refi


@pytest.mark.parametrize(
    "manifest_type", [RecordingSet, SupervisionSet, FeatureSet, CutSet]
)
def test_repeat_infinite(manifest_type):
    data = DummyManifest(manifest_type, begin_id=0, end_id=10)

    # hard to test infinite iterables, iterate it 10x more times than the original size
    eager_result = data.repeat()
    for idx, item in enumerate(eager_result):
        if idx == 105:
            break
    assert idx == 105

    with as_lazy(data) as lazy_data:
        lazy_result = lazy_data.repeat()
        for idx, item in enumerate(lazy_result):
            if idx == 105:
                break
        assert idx == 105


@pytest.mark.parametrize(
    "manifest_type", [RecordingSet, SupervisionSet, FeatureSet, CutSet]
)
def test_to_eager(manifest_type):
    data = DummyManifest(manifest_type, begin_id=0, end_id=10)

    with as_lazy(data) as lazy_data:
        eager_data = lazy_data.to_eager()
        assert isinstance(eager_data.data, type(data.data))
        assert eager_data == data
        assert list(eager_data) == list(data)


@pytest.mark.parametrize(
    "manifest_type",
    [
        RecordingSet,
        SupervisionSet,
        pytest.param(
            FeatureSet,
            marks=pytest.mark.xfail(reason="FeatureSet does not support shuffling."),
        ),
        CutSet,
    ],
)
def test_shuffle(manifest_type):
    data = DummyManifest(manifest_type, begin_id=0, end_id=4)
    for idx, item in enumerate(data):
        item.duration = idx

    expected_durations = [2, 1, 3, 0]

    rng = random.Random(42)

    eager_result = data.shuffle(rng=rng)
    assert [item.duration for item in eager_result] == list(expected_durations)

    with as_lazy(data) as lazy_data:
        lazy_result = lazy_data.shuffle(rng=rng)
        assert [item.duration for item in lazy_result] == list(expected_durations)


def test_composable_operations():
    expected_durations = [0, 2, 4, 6, 8, 0, 2, 4, 6, 8]

    data = DummyManifest(CutSet, begin_id=0, end_id=10)
    for idx, cut in enumerate(data):
        cut.duration = idx

    def less_than_5s(item):
        return item.duration < 5

    def double_duration(item):
        return fastcopy(item, duration=item.duration * 2)

    eager_result = data.repeat(2).filter(less_than_5s).map(double_duration)
    assert [c.duration for c in eager_result] == expected_durations

    with as_lazy(data) as lazy_data:
        lazy_result = lazy_data.repeat(2).filter(less_than_5s).map(double_duration)
        assert [item.duration for item in lazy_result] == list(expected_durations)


def _get_ids(cuts):
    return [cut.id for cut in cuts]


@pytest.mark.xfail(
    not is_module_available("dill"),
    reason="This test will fail when 'dill' module is not installed as it won't be able to pickle a lambda.",
    raises=AttributeError,
)
def test_dillable():
    cuts = DummyManifest(CutSet, begin_id=0, end_id=2)
    with as_lazy(cuts) as lazy_cuts:
        lazy_cuts = lazy_cuts.map(lambda c: fastcopy(c, id=c.id + "-random-suffix"))
        with ProcessPoolExecutor(1) as ex:
            # Moves the cutset which has a lambda stored somewhere to another process,
            # iterates it there, and gets results back to the main process.
            # Should work with dill, shouldn't work with just pickle.
            ids = list(ex.map(_get_ids, [lazy_cuts]))

        assert ids[0] == [
            "dummy-mono-cut-0000-random-suffix",
            "dummy-mono-cut-0001-random-suffix",
        ]
