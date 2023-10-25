from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from lhotse import (
    ChunkedLilcomHdf5Writer,
    LilcomChunkyWriter,
    LilcomFilesWriter,
    LilcomHdf5Writer,
    NumpyFilesWriter,
    NumpyHdf5Writer,
)
from lhotse.utils import is_module_available


@pytest.mark.parametrize(
    ["writer_type", "ext"],
    [
        (LilcomFilesWriter, ".llc"),
        (NumpyFilesWriter, ".npy"),
    ],
)
def test_writer_saved_file(writer_type, ext):
    # Generate small random numbers that are nicely compressed with lilcom
    arr = np.log(np.random.uniform(size=(11, 80)).astype(np.float32) / 100)

    with TemporaryDirectory() as d, writer_type(d) as writer:
        # testing that words after . is not replace
        input_key = "random0.3_vad.alpha"
        key = writer.write(input_key, arr)
        assert key == f"ran/{input_key}{ext}"

        # Testing when end with extension it is not added again
        input_key = f"temp0.2.alpha{ext}"
        key = writer.write(input_key, arr)
        assert key == f"tem/{input_key}"


@pytest.mark.parametrize(
    ["writer_type", "ext"],
    [
        pytest.param(
            NumpyHdf5Writer,
            ".h5",
            marks=pytest.mark.skipif(
                not is_module_available("h5py"),
                reason="Requires h5py to run HDF5 tests.",
            ),
        ),
        pytest.param(
            LilcomHdf5Writer,
            ".h5",
            marks=pytest.mark.skipif(
                not is_module_available("h5py"),
                reason="Requires h5py to run HDF5 tests.",
            ),
        ),
        pytest.param(
            ChunkedLilcomHdf5Writer,
            ".h5",
            marks=pytest.mark.skipif(
                not is_module_available("h5py"),
                reason="Requires h5py to run HDF5 tests.",
            ),
        ),
        (LilcomChunkyWriter, ".lca"),
    ],
)
def test_chunk_writer_saved_file(writer_type, ext):
    with TemporaryDirectory() as d:
        # testing that words after . is not replace
        filename = "random0.3_vad.alpha"
        with writer_type(f"{d}/{filename}") as writer:
            assert writer.storage_path_ == Path(f"{d}/{filename}{ext}")

        # Testing when end with extension it is not added again
        filename = f"random0.3_vad.alpha{ext}"
        with writer_type(f"{d}/{filename}") as writer:
            assert writer.storage_path_ == Path(f"{d}/{filename}")
