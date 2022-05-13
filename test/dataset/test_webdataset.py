import pickle
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pytest
from torch.utils.data import DataLoader

from lhotse import CutSet
from lhotse.dataset import DynamicCutSampler, IterableDatasetWrapper
from lhotse.dataset.webdataset import export_to_webdataset
from lhotse.utils import fastcopy

webdataset = pytest.importorskip(
    "webdataset", reason="These tests require webdataset package to run."
)


def test_export_to_webdataset():
    cuts = CutSet.from_file("test/fixtures/libri/cuts.json")
    cut = cuts[0]
    cuts = []
    for i in range(10):
        cuts.append(fastcopy(cut, id=cut.id + "-" + str(i)))
    cuts = CutSet.from_cuts(cuts)

    with NamedTemporaryFile(suffix=".tar") as f:
        export_to_webdataset(cuts, output_path=f.name)
        f.flush()

        ds = webdataset.WebDataset(f.name)

        dicts = (pickle.loads(data["data"]) for data in ds)

        cuts_ds = CutSet.from_dicts(dicts)

    assert list(cuts.ids) == list(cuts_ds.ids)


def test_cutset_from_webdataset():
    cuts = CutSet.from_file("test/fixtures/libri/cuts.json")
    cut = cuts[0]
    cuts = []
    for i in range(10):
        cuts.append(fastcopy(cut, id=cut.id + "-" + str(i)))
    cuts = CutSet.from_cuts(cuts)

    with NamedTemporaryFile(suffix=".tar") as f:
        export_to_webdataset(cuts, output_path=f.name)
        f.flush()

        cuts_ds = CutSet.from_webdataset(f.name)

        assert list(cuts.ids) == list(cuts_ds.ids)

        for c, cds in zip(cuts, cuts_ds):
            np.testing.assert_equal(c.load_audio(), cds.load_audio())
            np.testing.assert_almost_equal(
                c.load_features(), cds.load_features(), decimal=2
            )


def test_cutset_from_webdataset_sharded():
    cuts = CutSet.from_file("test/fixtures/libri/cuts.json")
    cut = cuts[0]
    cuts = []
    for i in range(10):
        cuts.append(fastcopy(cut, id=cut.id + "-" + str(i)))
    cuts = CutSet.from_cuts(cuts)

    with TemporaryDirectory() as dir_path:
        tar_pattern = f"{dir_path}/shard-%06d.tar"
        export_to_webdataset(cuts, output_path=tar_pattern, shard_size=2)

        # disabling shard shuffling for testing purposes here
        cuts_ds = CutSet.from_webdataset(
            dir_path + "/shard-{000000..000004}.tar", shuffle_shards=False
        )

        assert list(cuts.ids) == list(cuts_ds.ids)

        for c, cds in zip(cuts, cuts_ds):
            np.testing.assert_equal(c.load_audio(), cds.load_audio())
            np.testing.assert_almost_equal(
                c.load_features(), cds.load_features(), decimal=2
            )


def test_cutset_from_webdataset_sharded_pipe():
    cuts = CutSet.from_file("test/fixtures/libri/cuts.json")
    cut = cuts[0]
    cuts = []
    for i in range(10):
        cuts.append(fastcopy(cut, id=cut.id + "-" + str(i)))
    cuts = CutSet.from_cuts(cuts)

    with TemporaryDirectory() as dir_path:
        tar_pattern = f"pipe:gzip -c > {dir_path}/shard-%06d.tar.gz"
        export_to_webdataset(cuts, output_path=tar_pattern, shard_size=2)

        # disabling shard shuffling for testing purposes here
        cuts_ds = CutSet.from_webdataset(
            "pipe:gunzip -c " + dir_path + "/shard-{000000..000004}.tar.gz",
            shuffle_shards=False,
        )

        assert list(cuts.ids) == list(cuts_ds.ids)

        for c, cds in zip(cuts, cuts_ds):
            np.testing.assert_equal(c.load_audio(), cds.load_audio())
            np.testing.assert_almost_equal(
                c.load_features(), cds.load_features(), decimal=2
            )


class DummyDataset:
    """Dataset that returns input cuts."""

    def __getitem__(self, item):
        return item


def test_webdataset_sampler_epoch_increment():
    cuts = CutSet.from_file("test/fixtures/libri/cuts.json").repeat(10)

    with TemporaryDirectory() as dir_path:
        tar_pattern = f"{dir_path}/shard-%06d.tar"
        export_to_webdataset(cuts, output_path=tar_pattern, shard_size=1)

        cuts_ds = CutSet.from_webdataset(
            [str(p) for p in Path(dir_path).glob("*.tar")], shuffle_shards=True
        )
        sampler = DynamicCutSampler(cuts_ds, max_cuts=1)
        dloader = DataLoader(
            IterableDatasetWrapper(DummyDataset(), sampler, auto_increment_epoch=True),
            batch_size=None,
            num_workers=1,
            persistent_workers=True,
        )

        epoch_batches = {}
        for epoch in [0, 1]:
            batches = []
            for batch in dloader:
                for cut in batch:
                    batches.append(cut)
            epoch_batches[epoch] = CutSet.from_cuts(batches)

        # Both epochs have the same cut IDs.
        assert sorted(epoch_batches[0].ids) == sorted(epoch_batches[1].ids)
        # Both epochs have different cut order (shards were re-shuffled).
        assert list(epoch_batches[0].ids) != list(epoch_batches[1].ids)
