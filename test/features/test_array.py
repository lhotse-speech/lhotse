from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pytest

from lhotse import (
    ChunkedLilcomHdf5Writer,
    LilcomChunkyWriter,
    LilcomFilesWriter,
    LilcomHdf5Writer,
    MonoCut,
    NumpyFilesWriter,
    NumpyHdf5Writer,
)
from lhotse.array import Array
from lhotse.utils import is_module_available


@pytest.mark.parametrize(
    "array",
    [
        np.arange(20),
        np.arange(20).reshape(2, 10),
        np.arange(20).reshape(2, 5, 2),
        np.arange(20).astype(np.float32),
        np.arange(20).astype(np.int8),
    ],
)
@pytest.mark.parametrize(
    "writer_class",
    [
        NumpyFilesWriter,
        pytest.param(
            NumpyHdf5Writer,
            marks=pytest.mark.skipif(
                not is_module_available("h5py"),
                reason="Requires h5py to run HDF5 tests.",
            ),
        ),
        pytest.param(
            LilcomChunkyWriter,
            marks=pytest.mark.xfail(reason="Lilcom changes dtype to float32"),
        ),
        pytest.param(
            LilcomFilesWriter,
            marks=pytest.mark.xfail(reason="Lilcom changes dtype to float32"),
        ),
        pytest.param(
            LilcomHdf5Writer,
            marks=pytest.mark.xfail(reason="Lilcom changes dtype to float32"),
        ),
        pytest.param(
            ChunkedLilcomHdf5Writer,
            marks=pytest.mark.xfail(
                reason="Lilcom changes dtype to float32 (and Chunked variant works only with shape 2)"
            ),
        ),
    ],
)
def test_write_read_array_no_lilcom(array, writer_class):
    with TemporaryDirectory() as d, writer_class(d) as writer:
        manifest = writer.store_array(key="utt1", value=array)
        restored = manifest.load()
        assert array.ndim == manifest.ndim
        assert array.shape == restored.shape
        assert list(array.shape) == manifest.shape
        assert array.dtype == restored.dtype
        np.testing.assert_almost_equal(array, restored)


@pytest.mark.parametrize(
    "writer_class",
    [
        LilcomFilesWriter,
        pytest.param(
            LilcomHdf5Writer,
            marks=pytest.mark.skipif(
                not is_module_available("h5py"),
                reason="Requires h5py to run HDF5 tests.",
            ),
        ),
    ],
)
def test_write_read_array_lilcom(writer_class):
    array = np.arange(20).astype(np.float32)
    with TemporaryDirectory() as d, writer_class(d) as writer:
        manifest = writer.store_array(key="utt1", value=array)
        restored = manifest.load()
        assert array.ndim == manifest.ndim
        assert array.shape == restored.shape
        assert list(array.shape) == manifest.shape
        assert array.dtype == restored.dtype
        np.testing.assert_almost_equal(array, restored)


def test_array_serialization():
    # Individual items do not support JSON/etc. serialization;
    # instead, the XSet (e.g. CutSet) classes convert them to dicts.
    manifest = Array(
        storage_type="lilcom_hdf5",
        storage_path="/tmp/data",
        storage_key="irrelevant",
        shape=[300],
    )
    serialized = manifest.to_dict()
    restored = Array.from_dict(serialized)
    assert manifest == restored


def test_array_set_prefix_path():
    arr = Array(
        storage_type="lilcom_hdf5",
        storage_path="data/train",
        storage_key="irrelevant",
        shape=[300],
    )
    arr1 = arr.with_path_prefix("/newhome")
    assert arr1.storage_path == "/newhome/data/train"
    assert arr1.storage_type == arr.storage_type
    assert arr1.storage_key == arr.storage_key
    assert arr1.shape == arr.shape
