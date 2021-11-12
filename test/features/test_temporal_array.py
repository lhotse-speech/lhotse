from tempfile import TemporaryDirectory, NamedTemporaryFile

import numpy as np
import pytest

from lhotse import (
    ChunkedLilcomHdf5Writer,
    LilcomFilesWriter,
    LilcomHdf5Writer,
    NumpyFilesWriter,
    NumpyHdf5Writer,
)
from lhotse.array import Array, TemporalArray


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
def test_write_read_temporal_array_no_lilcom(array, writer_class):
    with TemporaryDirectory() as d, writer_class(d) as writer:
        manifest = writer.store_array(
            key="utt1",
            value=array,
            temporal_dim=0,
            frame_shift=0.4,
            start=0.0,
        )
        restored = manifest.load()
        assert array.ndim == manifest.ndim
        assert array.shape == restored.shape
        assert list(array.shape) == manifest.shape
        assert array.dtype == restored.dtype
        np.testing.assert_almost_equal(array, restored)


@pytest.mark.parametrize(
    "array",
    [
        np.arange(20).astype(np.float32),
        np.arange(20).reshape(2, 10).astype(np.float32),
        np.arange(20).reshape(2, 5, 2).astype(np.float32),
    ],
)
@pytest.mark.parametrize(
    "writer_class",
    [
        LilcomFilesWriter,
        LilcomHdf5Writer,
    ],
)
def test_write_read_temporal_array_lilcom(array, writer_class):
    with TemporaryDirectory() as d, writer_class(d) as writer:
        manifest = writer.store_array(
            key="utt1",
            value=array,
            temporal_dim=0,
            frame_shift=0.4,
            start=0.0,
        )
        restored = manifest.load()
        assert array.ndim == manifest.ndim
        assert array.shape == restored.shape
        assert list(array.shape) == manifest.shape
        assert array.dtype == restored.dtype
        np.testing.assert_almost_equal(array, restored)


def test_temporal_array_serialization():
    # Individual items do not support JSON/etc. serialization;
    # instead, the XSet (e.g. CutSet) classes convert them to dicts.
    manifest = TemporalArray(
        array=Array(
            storage_type="lilcom_hdf5",
            storage_path="/tmp/data",
            storage_key="irrelevant",
            shape=[300],
        ),
        temporal_dim=0,
        frame_shift=0.3,
        start=5.0,
    )
    serialized = manifest.to_dict()
    restored = TemporalArray.from_dict(serialized)
    assert manifest == restored


def test_temporal_array_partial_read():
    array = np.arange(30).astype(np.int8)

    with NamedTemporaryFile(suffix=".h5") as f, NumpyHdf5Writer(f.name) as writer:
        manifest = writer.store_array(
            key="utt1",
            value=array,
            temporal_dim=0,
            frame_shift=0.5,
            start=0.0,
        )

        # Read all
        restored = manifest.load()
        np.testing.assert_equal(array, restored)

        # Read first 10 frames (0 - 5 seconds)
        first_10 = manifest.load(duration=5)
        np.testing.assert_equal(array[:10], first_10)

        # Read last 10 frames (10 - 15 seconds)
        last_10 = manifest.load(start=10)
        np.testing.assert_equal(array[-10:], last_10)
        last_10 = manifest.load(start=10, duration=5)
        np.testing.assert_equal(array[-10:], last_10)

        # Read middle 10 frames (5 - 10 seconds)
        mid_10 = manifest.load(start=5, duration=5)
        np.testing.assert_equal(array[10:20], mid_10)
