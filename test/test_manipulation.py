from lhotse.utils import nullcontext as does_not_raise

import pytest
from pytest import mark, raises

from lhotse import CutSet
from lhotse.audio import RecordingSet
from lhotse.features import FeatureSet
from lhotse.manipulation import combine, load_manifest
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


@mark.parametrize(
    ['path', 'exception_expectation'],
    [
        ('test/fixtures/audio.json', does_not_raise()),
        ('test/fixtures/supervision.json', does_not_raise()),
        ('test/fixtures/dummy_feats/feature_manifest.json', does_not_raise()),
        ('test/fixtures/libri/cuts.json', does_not_raise()),
        ('test/fixtures/feature_config.yml', raises(ValueError)),
        ('no/such/path.xd', raises(FileNotFoundError)),
    ]
)
def test_load_any_lhotse_manifest(path, exception_expectation):
    with exception_expectation:
        load_manifest(path)
