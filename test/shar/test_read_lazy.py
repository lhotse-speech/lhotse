import os
import shutil
from functools import partial
from pathlib import Path

import numpy as np
import pytest

from lhotse import CutSet
from lhotse.shar import JsonlShardWriter, SharWriter
from lhotse.shar.readers.lazy import LazySharIterator


def test_shar_lazy_reader_from_dir(cuts: CutSet, shar_dir: Path):
    # Prepare system under test
    cuts_iter = LazySharIterator(in_dir=shar_dir)

    # Actual test
    for c_test, c_ref in zip(cuts_iter, cuts):
        assert c_test.id == c_ref.id
        assert c_test.has_custom("shard_origin")
        assert c_test.has_custom("shar_epoch")
        assert c_test.shar_epoch == 0
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


@pytest.mark.parametrize("shuffle", [True, False])
def test_shar_lazy_reader_shuffle(shar_dir: Path, shuffle: bool):
    reference = LazySharIterator(in_dir=shar_dir)
    # seed manually tweaked to get different order when shuffling 2 shards
    shuffled = LazySharIterator(in_dir=shar_dir, shuffle_shards=shuffle, seed=3)

    ref_paths = [cut.shard_origin for cut in reference]
    shf_paths = [cut.shard_origin for cut in shuffled]

    assert set(ref_paths) == set(shf_paths)
    assert len(ref_paths) == len(shf_paths)
    if shuffle:
        assert ref_paths != shf_paths  # different order
    else:
        assert ref_paths == shf_paths  # same order


@pytest.mark.parametrize("stateful", [True, False])
def test_shar_lazy_reader_shuffle_stateful(shar_dir: Path, stateful: bool):
    # seed 0 yields different shuffling in ep0 and ep1 for two shards
    shuffled = LazySharIterator(
        in_dir=shar_dir, shuffle_shards=True, stateful_shuffle=stateful, seed=0
    )

    ep0 = [c.shard_origin for c in shuffled]
    ep1 = [c.shard_origin for c in shuffled]

    assert set(ep0) == set(ep1)
    assert len(ep0) == len(ep1)
    if stateful:
        assert ep0 != ep1  # different order
    else:
        assert ep0 == ep1  # same order


def test_cut_set_from_shar(cuts: CutSet, shar_dir: Path):
    # Prepare system under test
    cuts_iter = CutSet.from_shar(in_dir=shar_dir)

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
        np.testing.assert_allclose(c_test.load_audio(), c_ref.load_audio())
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


def test_shar_lazy_reader_with_attached_attributes(cuts: CutSet, shar_dir: Path):

    # Create a custom attribute shards that we will dynamically attach
    cuts = cuts.from_shar(in_dir=shar_dir)
    with JsonlShardWriter(
        f"{shar_dir}/my_attributes.%06d.jsonl.gz", shard_size=10
    ) as w1, JsonlShardWriter(
        f"{shar_dir}/a_number.%06d.jsonl.gz", shard_size=10
    ) as w2:
        for idx, cut in enumerate(cuts):
            w1.write(
                {
                    "cut_id": cut.id,
                    "my_attributes": {
                        "attr1": idx,
                        "attr2": str(idx**2),
                    },
                }
            )
            if idx > 7:
                w2.write({"cut_id": cut.id, "a_number": 10})
            else:
                w2.write_placeholder(cut.id)

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
            "my_attributes": [
                shar_dir / "my_attributes.000000.jsonl.gz",
                shar_dir / "my_attributes.000001.jsonl.gz",
            ],
            "a_number": [
                shar_dir / "a_number.000000.jsonl.gz",
                shar_dir / "a_number.000001.jsonl.gz",
            ],
        }
    )

    # Actual test
    for idx, (c_test, c_ref) in enumerate(zip(cuts_iter, cuts)):
        assert c_test.id == c_ref.id

        # the recording is there
        np.testing.assert_allclose(c_ref.load_audio(), c_test.load_audio(), rtol=1e-3)

        # custom non-data attributes are attached
        attrs = c_test.my_attributes
        assert attrs["attr1"] == idx
        assert attrs["attr2"] == str(idx**2)
        if idx > 7:
            assert c_test.a_number == 10
        else:
            assert not c_test.has_custom("a_number")


@pytest.mark.parametrize("shuffle", [True, False])
def test_shar_lazy_reader_with_cut_map_fns(cuts: CutSet, shar_dir: Path, shuffle):
    # Prepare system under test

    def cut_map_fn_0(cut):
        cut.dataset = "dataset_corresponding_to_shard_0"
        return cut

    def cut_map_fn_1(cut):
        cut.dataset = "dataset_corresponding_to_shard_1"
        return cut

    cut_map_fns = [cut_map_fn_0, cut_map_fn_1]

    cuts_iter = LazySharIterator(
        fields={
            "cuts": [
                shar_dir / "cuts.000000.jsonl.gz",
                shar_dir / "cuts.000001.jsonl.gz",
            ],
        },
        cut_map_fns=cut_map_fns,
        shuffle_shards=shuffle,
    )

    # Actual test
    for c_test, c_ref in zip(cuts_iter, cuts):

        # when not shuffling, check the cut ID is correct;
        # when shuffling; check that the shuffle actually happened
        if shuffle:
            assert c_test.id != c_ref.id
        else:
            assert c_test.id == c_ref.id

        if c_test.shard_origin == shar_dir / "cuts.000000.jsonl.gz":
            assert c_test.dataset == "dataset_corresponding_to_shard_0"
        elif c_test.shard_origin == shar_dir / "cuts.000001.jsonl.gz":
            assert c_test.dataset == "dataset_corresponding_to_shard_1"
        else:
            raise RuntimeError(f"Unexpected shard_origin: {c_test.shard_origin}")


def test_shar_lazy_reader_from_fields_with_random_tar_name(
    cuts: CutSet, shar_dir: Path
):
    path0 = shar_dir / "random_name_y.tar"
    path1 = shar_dir / "random_another_name_x.tar"
    shutil.copy(shar_dir / "recording.000000.tar", path0)
    shutil.copy(shar_dir / "recording.000001.tar", path1)

    # Prepare system under test
    cuts_shar = CutSet.from_shar(
        fields={
            "cuts": [
                shar_dir / "cuts.000000.jsonl.gz",
                shar_dir / "cuts.000001.jsonl.gz",
            ],
            "recording": [path0, path1],
        }
    )

    # Actual test
    for c_test, c_ref in zip(cuts_shar, cuts):
        assert c_test.id == c_ref.id
        np.testing.assert_allclose(c_ref.load_audio(), c_test.load_audio(), rtol=1e-3)


def test_shar_lazy_reader_from_dir_multi_cuts(
    multi_cuts: CutSet, multi_cut_shar_dir: Path
):
    # Prepare system under test
    cuts_iter = LazySharIterator(in_dir=multi_cut_shar_dir)

    # Actual test
    for c_test, c_ref in zip(cuts_iter, multi_cuts):
        assert c_test.id == c_ref.id
        assert c_test.has_custom("shard_origin")
        assert c_test.has_custom("shar_epoch")
        assert c_test.shar_epoch == 0
        np.testing.assert_allclose(c_ref.load_audio(), c_test.load_audio(), rtol=1e-3)
        np.testing.assert_almost_equal(
            c_ref.load_features(), c_test.load_features(), decimal=1
        )
        np.testing.assert_equal(
            c_ref.load_custom_indexes(), c_test.load_custom_indexes()
        )


def test_shar_lazy_reader_from_dir_multi_cuts_multi_audio_source(
    multi_cuts_multi_audio_source: CutSet,
    multi_cut_multi_audio_source_shar_dir: Path,
):
    # Prepare system under test
    cuts_iter = LazySharIterator(in_dir=multi_cut_multi_audio_source_shar_dir)

    # Actual test
    for c_test, c_ref in zip(cuts_iter, multi_cuts_multi_audio_source):
        assert c_test.id == c_ref.id
        assert c_test.has_custom("shard_origin")
        assert c_test.has_custom("shar_epoch")
        assert c_test.shar_epoch == 0
        np.testing.assert_allclose(c_ref.load_audio(), c_test.load_audio(), rtol=1e-3)
        np.testing.assert_almost_equal(
            c_ref.load_features(), c_test.load_features(), decimal=1
        )
        np.testing.assert_equal(
            c_ref.load_custom_indexes(), c_test.load_custom_indexes()
        )


def test_shar_write_read_recordings_longer_than_cuts(cuts_from_long_recordings, tmpdir):
    tmpdir = Path(tmpdir)
    # Prepare data
    writer = SharWriter(
        tmpdir,
        fields={
            "recording": "wav",
            "features": "lilcom",
            "custom_features": "lilcom",
            "custom_indexes": "numpy",
            "custom_recording": "wav",
        },
        shard_size=10,
    )
    with writer:
        for c in cuts_from_long_recordings:
            writer.write(c)

    # Prepare system under test
    cuts_iter = LazySharIterator(in_dir=tmpdir)

    # Actual test
    for c_test, c_ref in zip(cuts_iter, cuts_from_long_recordings):
        assert c_test.id == c_ref.id
        assert c_test.has_custom("shard_origin")
        assert c_test.has_custom("shar_epoch")
        assert c_test.shar_epoch == 0
        np.testing.assert_allclose(c_ref.load_audio(), c_test.load_audio(), rtol=1e-3)
        np.testing.assert_almost_equal(
            c_ref.load_features(), c_test.load_features(), decimal=1
        )
        np.testing.assert_almost_equal(
            c_ref.load_custom_features(), c_test.load_custom_features(), decimal=1
        )
        np.testing.assert_equal(
            c_ref.load_custom_indexes(), c_test.load_custom_indexes()
        )
