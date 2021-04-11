import pytest
from pytest import mark

from lhotse import CutSet
from lhotse.audio import RecordingSet
from lhotse.features import FeatureSet
from lhotse.manipulation import combine
from lhotse.supervision import SupervisionSet
from lhotse.testing.dummies import DummyManifest


@mark.parametrize('manifest_type', [RecordingSet, SupervisionSet, FeatureSet, CutSet])
def test_split_even(manifest_type):
    manifest = DummyManifest(manifest_type, begin_id=0, end_id=100)
    manifest_subsets = manifest.split(num_splits=2)
    assert len(manifest_subsets) == 2
    assert manifest_subsets[0] == DummyManifest(manifest_type, begin_id=0, end_id=50)
    assert manifest_subsets[1] == DummyManifest(manifest_type, begin_id=50, end_id=100)


@mark.parametrize('manifest_type', [RecordingSet, SupervisionSet, FeatureSet, CutSet])
def test_split_randomize(manifest_type):
    manifest = DummyManifest(manifest_type, begin_id=0, end_id=100)
    manifest_subsets = manifest.split(num_splits=2, shuffle=True)
    assert len(manifest_subsets) == 2
    recombined_items = list(manifest_subsets[0]) + list(manifest_subsets[1])
    assert len(recombined_items) == len(manifest)
    # Different ordering (we convert to lists first because the *Set classes might internally
    # re-order after concatenation, e.g. by using dict or post-init sorting)
    assert recombined_items != list(manifest)


@mark.parametrize('manifest_type', [RecordingSet, SupervisionSet, FeatureSet, CutSet])
def test_split_odd(manifest_type):
    manifest = DummyManifest(manifest_type, begin_id=0, end_id=100)
    manifest_subsets = manifest.split(num_splits=3)
    assert len(manifest_subsets) == 3
    assert manifest_subsets[0] == DummyManifest(manifest_type, begin_id=0, end_id=34)
    assert manifest_subsets[1] == DummyManifest(manifest_type, begin_id=34, end_id=68)
    assert manifest_subsets[2] == DummyManifest(manifest_type, begin_id=68, end_id=100)


@mark.parametrize('manifest_type', [RecordingSet, SupervisionSet, FeatureSet, CutSet])
def test_cannot_split_to_more_chunks_than_items(manifest_type):
    manifest = DummyManifest(manifest_type, begin_id=0, end_id=1)
    with pytest.raises(ValueError):
        manifest.split(num_splits=10)


@mark.parametrize('manifest_type', [RecordingSet, SupervisionSet, FeatureSet, CutSet])
def test_combine(manifest_type):
    expected = DummyManifest(manifest_type, begin_id=0, end_id=200)
    combined = combine(
        DummyManifest(manifest_type, begin_id=0, end_id=68),
        DummyManifest(manifest_type, begin_id=68, end_id=136),
        DummyManifest(manifest_type, begin_id=136, end_id=200),
    )
    assert combined == expected
    combined_iterable = combine([
        DummyManifest(manifest_type, begin_id=0, end_id=68),
        DummyManifest(manifest_type, begin_id=68, end_id=136),
        DummyManifest(manifest_type, begin_id=136, end_id=200),
    ])
    assert combined_iterable == expected


@mark.parametrize('manifest_type', [RecordingSet, SupervisionSet, FeatureSet, CutSet])
def test_subset_first(manifest_type):
    any_set = DummyManifest(manifest_type, begin_id=0, end_id=200)
    expected = DummyManifest(manifest_type, begin_id=0, end_id=10)
    subset = any_set.subset(first=10)
    assert subset == expected


@mark.parametrize('manifest_type', [RecordingSet, SupervisionSet, FeatureSet, CutSet])
def test_subset_last(manifest_type):
    any_set = DummyManifest(manifest_type, begin_id=0, end_id=200)
    expected = DummyManifest(manifest_type, begin_id=190, end_id=200)
    subset = any_set.subset(last=10)
    assert subset == expected


@mark.parametrize('manifest_type', [RecordingSet, SupervisionSet, FeatureSet, CutSet])
@mark.parametrize(['first', 'last'], [(None, None), (10, 10)])
def test_subset_raises(manifest_type, first, last):
    any_set = DummyManifest(manifest_type, begin_id=0, end_id=200)
    with pytest.raises(AssertionError):
        subset = any_set.subset(first=first, last=last)
