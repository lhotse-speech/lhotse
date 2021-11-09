from tempfile import NamedTemporaryFile

import numpy as np

from lhotse import LilcomHdf5Writer, MonoCut
from lhotse.array import Array


def test_cut_load_array():
    array = np.arange(20).astype(np.float32)
    with NamedTemporaryFile(suffix=".h5") as f, LilcomHdf5Writer(f.name) as writer:
        manifest = Array.store(key="utt1", value=array, writer=writer)
        cut = MonoCut(id="x", start=0, duration=5, channel=0)
        # Note: MonoCut doesn't normally have an "ivector" attribute,
        #       and a "load_ivector()" method.
        #       We are dynamically extending it.
        cut.ivector = manifest
        restored = cut.load_ivector()
        np.testing.assert_equal(array, restored)
