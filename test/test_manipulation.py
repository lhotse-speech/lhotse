import random
from tempfile import TemporaryDirectory

import pytest
from pytest import mark

from lhotse import CutSet
from lhotse.audio import RecordingSet
from lhotse.features import FeatureSet
from lhotse.manipulation import combine
from lhotse.supervision import SupervisionSet
from lhotse.testing.dummies import DummyManifest, as_lazy
from lhotse.utils import nullcontext


@mark.parametrize("manifest_type", [RecordingSet, SupervisionSet, FeatureSet, CutSet])
def test_split_even(manifest_type):
    manifest = DummyManifest(manifest_type, begin_id=0, end_id=100)
    manifest_subsets = manifest.split(num_splits=2)
    assert len(manifest_subsets) == 2
    assert manifest_subsets[0] == DummyManifest(manifest_type, begin_id=0, end_id=50)
    assert manifest_subsets[1] == DummyManifest(manifest_type, begin_id=50, end_id=100)


@mark.parametrize("manifest_type", [RecordingSet, SupervisionSet, FeatureSet, CutSet])
def test_split_randomize(manifest_type):
    manifest = DummyManifest(manifest_type, begin_id=0, end_id=100)
    manifest_subsets = manifest.split(num_splits=2, shuffle=True)
    assert len(manifest_subsets) == 2
    recombined_items = list(manifest_subsets[0]) + list(manifest_subsets[1])
    assert len(recombined_items) == len(manifest)
    # Different ordering (we convert to lists first because the *Set classes might internally
    # re-order after concatenation, e.g. by using dict or post-init sorting)
    assert recombined_items != list(manifest)


@mark.parametrize("manifest_type", [RecordingSet, SupervisionSet, FeatureSet, CutSet])
@mark.parametrize("drop_last", [True, False])
def test_split_odd_1(manifest_type, drop_last):
    manifest = DummyManifest(manifest_type, begin_id=0, end_id=100)
    manifest_subsets = manifest.split(num_splits=3, drop_last=drop_last)
    assert len(manifest_subsets) == 3
    if drop_last:
        assert manifest_subsets[0] == DummyManifest(
            manifest_type, begin_id=0, end_id=33
        )
        assert manifest_subsets[1] == DummyManifest(
            manifest_type, begin_id=33, end_id=66
        )
        assert manifest_subsets[2] == DummyManifest(
            manifest_type, begin_id=66, end_id=99
        )
    else:
        assert manifest_subsets[0] == DummyManifest(
            manifest_type, begin_id=0, end_id=34
        )
        assert manifest_subsets[1] == DummyManifest(
            manifest_type, begin_id=34, end_id=67
        )
        assert manifest_subsets[2] == DummyManifest(
            manifest_type, begin_id=67, end_id=100
        )


@mark.parametrize("manifest_type", [RecordingSet, SupervisionSet, FeatureSet, CutSet])
@mark.parametrize("drop_last", [True, False])
def test_split_odd_2(manifest_type, drop_last):
    manifest = DummyManifest(manifest_type, begin_id=0, end_id=32)
    manifest_subsets = manifest.split(num_splits=3, drop_last=drop_last)
    assert len(manifest_subsets) == 3
    if drop_last:
        assert manifest_subsets[0] == DummyManifest(
            manifest_type, begin_id=0, end_id=10
        )
        assert manifest_subsets[1] == DummyManifest(
            manifest_type, begin_id=10, end_id=20
        )
        assert manifest_subsets[2] == DummyManifest(
            manifest_type, begin_id=20, end_id=30
        )
    else:
        assert manifest_subsets[0] == DummyManifest(
            manifest_type, begin_id=0, end_id=11
        )
        assert manifest_subsets[1] == DummyManifest(
            manifest_type, begin_id=11, end_id=22
        )
        assert manifest_subsets[2] == DummyManifest(
            manifest_type, begin_id=22, end_id=32
        )


@mark.parametrize("manifest_type", [RecordingSet, SupervisionSet, FeatureSet, CutSet])
def test_cannot_split_to_more_chunks_than_items(manifest_type):
    manifest = DummyManifest(manifest_type, begin_id=0, end_id=1)
    with pytest.raises(ValueError):
        manifest.split(num_splits=10)


@mark.parametrize("manifest_type", [RecordingSet, SupervisionSet, FeatureSet, CutSet])
def test_split_lazy_even(manifest_type):
    with TemporaryDirectory() as d:
        manifest = DummyManifest(manifest_type, begin_id=0, end_id=100)
        manifest_subsets = manifest.split_lazy(output_dir=d, chunk_size=49)
        assert len(manifest_subsets) == 3
        assert list(manifest_subsets[0]) == list(
            DummyManifest(manifest_type, begin_id=0, end_id=49)
        )
        assert list(manifest_subsets[1]) == list(
            DummyManifest(manifest_type, begin_id=49, end_id=98)
        )
        assert list(manifest_subsets[2]) == list(
            DummyManifest(manifest_type, begin_id=98, end_id=100)
        )


def test_split_lazy_edge_case_extra_shard(tmp_path):
    N = 512
    chsz = 32
    nshrd = 16
    manifest = DummyManifest(CutSet, begin_id=0, end_id=N - 1)
    manifest_subsets = manifest.split_lazy(output_dir=tmp_path, chunk_size=chsz)
    assert len(manifest_subsets) == nshrd
    for item in sorted(tmp_path.glob("*")):
        print(item)


@mark.parametrize("manifest_type", [RecordingSet, SupervisionSet, FeatureSet, CutSet])
def test_combine(manifest_type):
    expected = DummyManifest(manifest_type, begin_id=0, end_id=200)
    combined = combine(
        DummyManifest(manifest_type, begin_id=0, end_id=68),
        DummyManifest(manifest_type, begin_id=68, end_id=136),
        DummyManifest(manifest_type, begin_id=136, end_id=200),
    )
    assert combined.to_eager() == expected
    combined_iterable = combine(
        [
            DummyManifest(manifest_type, begin_id=0, end_id=68),
            DummyManifest(manifest_type, begin_id=68, end_id=136),
            DummyManifest(manifest_type, begin_id=136, end_id=200),
        ]
    )
    assert combined_iterable.to_eager() == expected


@mark.parametrize("manifest_type", [RecordingSet, SupervisionSet, FeatureSet, CutSet])
@mark.parametrize("lazy", [False, True])
def test_subset_first(manifest_type, lazy):
    ctx = as_lazy if lazy else nullcontext
    any_set = DummyManifest(manifest_type, begin_id=0, end_id=200)
    with ctx(any_set, ".jsonl") as any_set:
        expected = DummyManifest(manifest_type, begin_id=0, end_id=10)
        subset = any_set.subset(first=10)
        assert subset == expected


@mark.parametrize("manifest_type", [RecordingSet, SupervisionSet, FeatureSet, CutSet])
@mark.parametrize("lazy", [False, True])
def test_subset_last(manifest_type, lazy):
    ctx = as_lazy if lazy else nullcontext
    any_set = DummyManifest(manifest_type, begin_id=0, end_id=200)
    with ctx(any_set, ".jsonl") as any_set:
        expected = DummyManifest(manifest_type, begin_id=190, end_id=200)
        subset = any_set.subset(last=10)
        assert subset == expected


@mark.parametrize("manifest_type", [RecordingSet, SupervisionSet, FeatureSet, CutSet])
@mark.parametrize(["first", "last"], [(None, None), (10, 10)])
def test_subset_raises(manifest_type, first, last):
    any_set = DummyManifest(manifest_type, begin_id=0, end_id=200)
    with pytest.raises(AssertionError):
        subset = any_set.subset(first=first, last=last)


@mark.parametrize("manifest_type", [RecordingSet, SupervisionSet, CutSet])
@mark.parametrize("rng", [None, random.Random(1337)])
def test_shuffle(manifest_type, rng):
    any_set = DummyManifest(manifest_type, begin_id=0, end_id=200)
    shuffled = any_set.shuffle(rng=rng)
    assert list(any_set.ids) != list(shuffled.ids)
    assert set(any_set.ids) == set(shuffled.ids)
