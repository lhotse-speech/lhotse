from pathlib import Path

import numpy as np
import pytest

from lhotse import CutSet
from lhotse.shar.readers.datapipes import load_shar_datapipe

dp = pytest.importorskip("torchdata")


@pytest.mark.skip(reason="We don't support datapipes for now.")
def test_shar_datapipe_reader(cuts: CutSet, shar_dir: Path):
    # Prepare system under test
    cuts_iter = load_shar_datapipe(shar_dir)

    # Actual test
    for c_test, c_ref in zip(cuts_iter, cuts):
        assert c_test.id == c_ref.id
        np.testing.assert_allclose(c_ref.load_audio(), c_test.load_audio(), rtol=1e-3)
        np.testing.assert_allclose(
            c_ref.load_custom_recording(), c_test.load_custom_recording(), rtol=1e-3
        )
        np.testing.assert_almost_equal(
            c_ref.load_features(), c_test.load_features(), decimal=1
        )
        np.testing.assert_almost_equal(
            c_ref.load_custom_features(), c_test.load_custom_features(), decimal=1
        )
        np.testing.assert_almost_equal(
            c_ref.load_custom_embedding(), c_test.load_custom_embedding(), decimal=1
        )
        np.testing.assert_almost_equal(
            c_ref.load_custom_indexes(), c_test.load_custom_indexes(), decimal=1
        )
