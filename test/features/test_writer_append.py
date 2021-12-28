from tempfile import TemporaryDirectory

import numpy as np
import pytest

from lhotse import CutSet, LilcomChunkyReader, LilcomChunkyWriter


@pytest.fixture
def cuts():
    return CutSet.from_file("test/fixtures/libri/cuts.json")


def test_writer_overwrite(cuts):
    data = cuts[0].load_features()
    dataplus1 = data + 1
    with TemporaryDirectory() as d:

        with LilcomChunkyWriter(d) as w:
            key1 = w.write('key1', data)
            storage_path = w.storage_path

        with LilcomChunkyWriter(storage_path) as w:
            key2 = w.write('key2', dataplus1)

        r = LilcomChunkyReader(storage_path)

        # key1 is corrupted as it was overwritten
        with pytest.raises(ValueError):
            r.read(key1)

        restored = r.read(key2)
        # decimal=1 because of lilcom compression, see:
        # E           AssertionError:
        # E           Arrays are not almost equal to 7 decimals
        # E
        # E           Mismatched elements: 39997 / 40000 (100%)
        # E           Max absolute difference: 0.015625
        # E           Max relative difference: 54.764027
        np.testing.assert_almost_equal(restored, dataplus1, decimal=1)


def test_writer_append(cuts):
    data = cuts[0].load_features()
    dataplus1 = data + 1
    with TemporaryDirectory() as d:

        with LilcomChunkyWriter(d) as w:
            key1 = w.write('key1', data)
            storage_path = w.storage_path

        with LilcomChunkyWriter(storage_path, mode='ab') as w:
            key2 = w.write('key2', dataplus1)

        r = LilcomChunkyReader(storage_path)

        restored = r.read(key1)
        np.testing.assert_almost_equal(restored, data, decimal=1)
        # decimal=1 because of lilcom compression, see:
        # E           AssertionError:
        # E           Arrays are not almost equal to 7 decimals
        # E
        # E           Mismatched elements: 39499 / 40000 (98.7%)
        # E           Max absolute difference: 0.015625
        # E           Max relative difference: 4.2855635

        restored = r.read(key2)
        # decimal=1 because of lilcom compression, see:
        # E           AssertionError:
        # E           Arrays are not almost equal to 7 decimals
        # E
        # E           Mismatched elements: 39997 / 40000 (100%)
        # E           Max absolute difference: 0.015625
        # E           Max relative difference: 54.764027
        np.testing.assert_almost_equal(restored, dataplus1, decimal=1)


