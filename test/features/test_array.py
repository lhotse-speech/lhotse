from tempfile import TemporaryDirectory

import numpy as np
import pytest

from lhotse import (
    ChunkedLilcomHdf5Writer,
    LilcomFilesWriter,
    LilcomHdf5Writer,
    NumpyFilesWriter,
    NumpyHdf5Writer,
)
from lhotse.array import Array


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
        NumpyHdf5Writer,
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
        manifest = Array.store(key="utt1", value=array, writer=writer)
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
        LilcomHdf5Writer,
    ],
)
def test_write_read_array_lilcom(writer_class):
    array = np.arange(20).astype(np.float32)
    with TemporaryDirectory() as d, writer_class(d) as writer:
        manifest = Array.store(key="utt1", value=array, writer=writer)
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
        storage_type='lilcom_hdf5',
        storage_path='/tmp/data',
        storage_key='irrelevant',
        shape=[300],
    )
    serialized = manifest.to_dict()
    restored = Array.from_dict(serialized)
    assert manifest == restored
