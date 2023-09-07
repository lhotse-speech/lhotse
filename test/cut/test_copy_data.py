from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pytest

from lhotse import CutSet, FeatureSet, NumpyFilesWriter
from lhotse.utils import fastcopy


@pytest.fixture
def cuts():
    return CutSet.from_file("test/fixtures/libri/cuts.json")


def test_copy_data(cuts):
    cuts = CutSet.from_cuts(
        [
            # MonoCut
            cuts[0],
            # MonoCut without feats
            fastcopy(cuts[0], id="cut-no-feats").drop_features(),
            # MonoCut without recording
            fastcopy(cuts[0], id="cut-no-rec").drop_recording(),
        ]
    )
    with TemporaryDirectory() as d:
        cpy = cuts.copy_data(d)
        assert len(cpy) == len(cuts)

        cut, ref = cpy[0], cuts[0]
        assert cut.id == ref.id
        assert cut.duration == ref.duration
        assert cut.has_features and ref.has_features
        # lilcom absolute tolerance
        np.testing.assert_allclose(cut.load_features(), ref.load_features(), atol=2e-2)
        assert cut.has_recording and ref.has_recording
        np.testing.assert_almost_equal(cut.load_audio(), ref.load_audio())

        cut, ref = cpy[1], cuts[1]
        assert cut.id == ref.id
        assert cut.duration == ref.duration
        assert not cut.has_features and not ref.has_features
        assert cut.has_recording and ref.has_recording

        cut, ref = cpy[2], cuts[2]
        assert cut.id == ref.id
        assert cut.duration == ref.duration
        assert cut.has_features and ref.has_features
        assert not cut.has_recording and not ref.has_recording


def test_cut_set_mixed_cut_copy_data(cuts):
    cuts = CutSet.from_cuts(
        [
            # MixedCut
            cuts[0].pad(duration=30)
        ]
    )
    with TemporaryDirectory() as d:
        cpy = cuts.copy_data(d)
        assert len(cpy) == len(cuts)

        cut, ref = cpy[0], cuts[0]
        assert cut.id == ref.id
        assert cut.duration == ref.duration
        assert cut.has_features and ref.has_features
        np.testing.assert_almost_equal(cut.load_features(), ref.load_features())
        assert cut.has_recording and ref.has_recording
        np.testing.assert_almost_equal(cut.load_audio(), ref.load_audio())
