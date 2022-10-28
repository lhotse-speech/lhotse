import os
from pathlib import Path

import numpy as np
import pytest

from lhotse import CutSet
from lhotse.shar.readers.lazy import LazySharIterator


def test_shar_lazy_reader_from_dir(cuts: CutSet, shar_dir: Path):
    # Prepare system under test
    cuts_iter = LazySharIterator(in_dir=shar_dir)

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


def test_shar_lazy_reader_from_fields(cuts: CutSet, shar_dir: Path):
    # Prepare system under test
    cuts_iter = LazySharIterator(
        fields={
            "cuts": [
                shar_dir / "cuts.000000.jsonl.gz",
                shar_dir / "cuts.000001.jsonl.gz",
            ],
            "recording": [
                shar_dir / "recording.000000.tar",
                shar_dir / "recording.000001.tar",
            ],
        }
    )

    # Actual test
    for c_test, c_ref in zip(cuts_iter, cuts):
        assert c_test.id == c_ref.id
        np.testing.assert_allclose(c_ref.load_audio(), c_test.load_audio(), rtol=1e-3)


@pytest.mark.skipif(os.name == "nt", reason="This test cannot run on Windows.")
def test_shar_lazy_reader_from_fields_using_pipes(cuts: CutSet, shar_dir: Path):
    # Prepare system under test
    cuts_iter = LazySharIterator(
        fields={
            "cuts": [
                f"pipe:gunzip -c {shar_dir}/cuts.000000.jsonl.gz",
                f"pipe:gunzip -c {shar_dir}/cuts.000001.jsonl.gz",
            ],
            "recording": [
                f"pipe:cat {shar_dir}/recording.000000.tar",
                f"pipe:cat {shar_dir}/recording.000001.tar",
            ],
        }
    )

    # Actual test
    for c_test, c_ref in zip(cuts_iter, cuts):
        assert c_test.id == c_ref.id
        np.testing.assert_allclose(c_ref.load_audio(), c_test.load_audio(), rtol=1e-3)


def test_shar_lazy_reader_raises_error_on_missing_field(cuts: CutSet, shar_dir: Path):
    # Prepare system under test
    cuts_iter = LazySharIterator(
        fields={
            "cuts": [
                shar_dir / "cuts.000000.jsonl.gz",
                shar_dir / "cuts.000001.jsonl.gz",
            ],
            "recording": [
                shar_dir / "recording.000000.tar",
                shar_dir / "recording.000001.tar",
            ],
        }
    )

    # Actual test
    for c_test, c_ref in zip(cuts_iter, cuts):
        assert c_test.id == c_ref.id
        with pytest.raises(RuntimeError):
            c_test.load_custom_recording()
        with pytest.raises(RuntimeError):
            c_test.load_features()
        with pytest.raises(RuntimeError):
            c_test.load_custom_features()
        with pytest.raises(RuntimeError):
            c_test.load_custom_indexes()
        with pytest.raises(RuntimeError):
            c_test.load_custom_embedding()
