from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pytest

from lhotse import CutSet, FeatureSet, NumpyFilesWriter
from lhotse.utils import fastcopy


@pytest.fixture
def cuts():
    return CutSet.from_file("test/fixtures/libri/cuts.json")


def test_features_copy_feats(cuts):
    features = cuts[0].features
    with TemporaryDirectory() as d, NumpyFilesWriter(d) as w:
        cpy = features.copy_feats(writer=w)
        data = cpy.load()
        assert isinstance(data, np.ndarray)
        ref_data = features.load()
        np.testing.assert_almost_equal(data, ref_data)


def test_feature_set_copy_feats(cuts):
    feature_set = FeatureSet.from_features([cuts[0].features])
    with TemporaryDirectory() as d, NumpyFilesWriter(d) as w:
        cpy = feature_set.copy_feats(writer=w)
        data = cpy[0].load()
        assert isinstance(data, np.ndarray)
        ref_data = feature_set[0].load()
        np.testing.assert_almost_equal(data, ref_data)


def test_cut_set_copy_feats(cuts):
    # Make a CutSet with MonoCut, PaddingCut, and MixedCut
    cuts = CutSet.from_cuts(
        [
            # MonoCut
            cuts[0],
            # MonoCut without feats
            fastcopy(cuts[0], id="cut-no-feats").drop_features(),
        ]
    )
    with TemporaryDirectory() as d, NumpyFilesWriter(d) as w:
        cpy = cuts.copy_feats(writer=w)
        assert len(cpy) == len(cuts)
        for cut, orig in zip(cpy, cuts):
            if not orig.has_features:
                continue
            data = cut.load_features()
            assert isinstance(data, np.ndarray)
            ref_data = orig.load_features()
            np.testing.assert_almost_equal(data, ref_data)


def test_cut_set_copy_feats_output_path(cuts):
    # Make a CutSet with MonoCut, PaddingCut, and MixedCut
    cuts = CutSet.from_cuts(
        [
            # MonoCut
            cuts[0],
            # MonoCut without feats
            fastcopy(cuts[0], id="cut-no-feats").drop_features(),
        ]
    )
    with NamedTemporaryFile(
        suffix=".jsonl"
    ) as f, TemporaryDirectory() as d, NumpyFilesWriter(d) as w:
        cpy = cuts.copy_feats(writer=w, output_path=f.name)
        assert len(cpy) == len(cuts)
        assert list(cpy.ids) == list(cuts.ids)
        for cut, orig in zip(cpy, cuts):
            assert (
                not orig.has_features
                or (cut.load_features() == orig.load_features()).all()
            )


def test_cut_set_mixed_cut_copy_feats(cuts):
    # Make a CutSet with MonoCut, PaddingCut, and MixedCut
    cuts = CutSet.from_cuts(
        [
            # MixedCut
            cuts[0].pad(duration=30)
        ]
    )
    with TemporaryDirectory() as d, NumpyFilesWriter(d) as w:
        cpy = cuts.copy_feats(writer=w)
        assert len(cpy) == len(cuts)
        for cut, orig in zip(cpy, cuts):
            data = cut.load_features()
            assert isinstance(data, np.ndarray)
            ref_data = orig.load_features()
            np.testing.assert_almost_equal(data, ref_data)
