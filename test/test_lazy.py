"""
Since the tests in this module handle very different types of *Set classes,
we try to leverage 'duration' attribute which is shared by all tested types of items
(cuts, features, recordings, supervisions).
"""
import random

import pytest

from lhotse import CutSet, FeatureSet, RecordingSet, SupervisionSet, combine
from lhotse.testing.dummies import DummyManifest, as_lazy
from lhotse.utils import fastcopy


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
        assert list(lazy_result) == list(expected)


@pytest.mark.parametrize(
    "manifest_type", [RecordingSet, SupervisionSet, FeatureSet, CutSet]
)
def test_repeat(manifest_type):
    data = DummyManifest(manifest_type, begin_id=0, end_id=10)

    expected = data + data

    eager_result = data.repeat(times=2)
    assert list(eager_result) == list(expected)

    with as_lazy(data) as lazy_data:
        lazy_result = lazy_data.repeat(times=2)
        assert list(lazy_result) == list(expected)


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
