from contextlib import nullcontext as does_not_raise

import pytest
from pytest import mark, raises

from lhotse.audio import RecordingSet
from lhotse.features import FeatureSet
from lhotse.manipulation import combine, load_manifest, split
from lhotse.supervision import SupervisionSet
from lhotse.test_utils import DummyManifest


@mark.parametrize('manifest_type', [RecordingSet, SupervisionSet, FeatureSet])
def test_split_even(manifest_type):
    manifest = DummyManifest(manifest_type, begin_id=0, end_id=100)
    manifest_subsets = split(manifest, num_splits=2)
    assert len(manifest_subsets) == 2
    assert manifest_subsets[0] == DummyManifest(manifest_type, begin_id=0, end_id=50)
    assert manifest_subsets[1] == DummyManifest(manifest_type, begin_id=50, end_id=100)


@mark.parametrize('manifest_type', [RecordingSet, SupervisionSet, FeatureSet])
def test_split_odd(manifest_type):
    manifest = DummyManifest(manifest_type, begin_id=0, end_id=100)
    manifest_subsets = split(manifest, num_splits=3)
    assert len(manifest_subsets) == 3
    assert manifest_subsets[0] == DummyManifest(manifest_type, begin_id=0, end_id=34)
    assert manifest_subsets[1] == DummyManifest(manifest_type, begin_id=34, end_id=68)
    assert manifest_subsets[2] == DummyManifest(manifest_type, begin_id=68, end_id=100)


@mark.parametrize('manifest_type', [RecordingSet, SupervisionSet, FeatureSet])
def test_cannot_split_to_more_chunks_than_items(manifest_type):
    manifest = DummyManifest(manifest_type, begin_id=0, end_id=1)
    with pytest.raises(ValueError):
        split(manifest, num_splits=10)


@mark.parametrize('manifest_type', [RecordingSet, SupervisionSet, FeatureSet])
def test_combine(manifest_type):
    expected = DummyManifest(manifest_type, begin_id=0, end_id=200)
    combined = combine(
        DummyManifest(manifest_type, begin_id=0, end_id=68),
        DummyManifest(manifest_type, begin_id=68, end_id=136),
        DummyManifest(manifest_type, begin_id=136, end_id=200),
    )
    assert combined == expected


@mark.parametrize(
    ['path', 'exception_expectation'],
    [
        ('test/fixtures/audio.json', does_not_raise()),
        ('test/fixtures/supervision.json', does_not_raise()),
        ('test/fixtures/dummy_feats/feature_manifest.json', does_not_raise()),
        ('test/fixtures/feature_config.yml', raises(ValueError)),
        ('no/such/path.xd', raises(FileNotFoundError)),
    ]
)
def test_load_any_lhotse_manifest(path, exception_expectation):
    with exception_expectation:
        load_manifest(path)
