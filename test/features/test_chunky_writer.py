from tempfile import NamedTemporaryFile

import numpy as np
import pytest

from lhotse import ChunkedLilcomHdf5Writer, LilcomChunkyWriter
from lhotse.features.io import get_reader


@pytest.mark.parametrize(
    ["writer_type", "ext"],
    [(LilcomChunkyWriter, ".lca"), (ChunkedLilcomHdf5Writer, ".h5")],
)
def test_chunky_writer_left_right_offsets_equal(writer_type, ext):
    # Generate small random numbers that are nicely compressed with lilcom
    arr = np.log(np.random.uniform(size=(11, 80)).astype(np.float32) / 100)

    with NamedTemporaryFile(suffix=ext) as f:

        with writer_type(f.name) as writer:
            key = writer.write("dummy-key", arr)

        f.flush()
        reader = get_reader(writer.name)(f.name)

        # Reading full array -- works as expected
        arr1 = reader.read(key)
        np.testing.assert_almost_equal(arr, arr1, decimal=1)

        # Reading an empty subset should return an empty array
        arr2 = reader.read(key, left_offset_frames=0, right_offset_frames=0)
        assert arr2.shape == (0,)
