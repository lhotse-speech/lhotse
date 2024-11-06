import re
from collections import Counter
from pathlib import Path
from typing import Tuple

import pytest
import torch.utils.data

from lhotse import CutSet
from lhotse.dataset import CutConcatenate, IterableDatasetWrapper
from lhotse.dataset.sampling import StatelessSampler
from lhotse.testing.dummies import DummyManifest, as_lazy


@pytest.fixture()
def cuts_files(tmp_path_factory) -> Tuple[Path, Path]:
    tmp_dir = tmp_path_factory.getbasetemp()
    paths = (tmp_dir / "cuts_A.jsonl", tmp_dir / "cuts_B.jsonl")
    cs = DummyManifest(CutSet, begin_id=0, end_id=10)
    cs.to_file(paths[0])
    cs = DummyManifest(CutSet, begin_id=10, end_id=20)
    for c in cs:
        c.duration = 2  # for bucketing tests
    cs.to_file(paths[1])
    return paths


def test_stateless_sampler_single_file(cuts_files: Tuple[Path]):
    path = cuts_files[0]
    index_path = cuts_files[0].parent / "cuts.idx"
    sampler = StatelessSampler(path, index_path=index_path, max_cuts=2, base_seed=0)

    for idx, batch in enumerate(sampler):
        assert len(batch) == 2
        for cut in batch:
            # Assert the cut IDs do not go into two digit range
            # (because cuts_files[0] has only 0-9)
            assert re.match(r"dummy-mono-cut-000\d.*", cut.id) is not None, cut.id
        if idx == 4:
            break  # the sampler is infinite

    assert sampler.diagnostics.total_batches == 5
    assert sampler.diagnostics.total_cuts == 10

    assert index_path.is_file()
    assert cuts_files[0].with_suffix(".jsonl.idx").is_file()


def test_stateless_sampler_multi_files(cuts_files: Tuple[Path]):
    index_path = cuts_files[0].parent / "cuts.idx"
    sampler = StatelessSampler(
        cuts_files, index_path=index_path, max_cuts=2, base_seed=0
    )

    for idx, batch in enumerate(sampler):
        assert len(batch) == 2
        for cut in batch:
            # The cut IDs will be in range 0-19,
            # with first file having 0-9 and second 10-19.
            assert re.match(r"dummy-mono-cut-00[01]\d.*", cut.id) is not None, cut.id
        if idx == 4:
            break  # the sampler is infinite

    assert sampler.diagnostics.total_batches == 5
    assert sampler.diagnostics.total_cuts == 10

    assert index_path.is_file()
    assert cuts_files[0].with_suffix(".jsonl.idx").is_file()
    assert cuts_files[1].with_suffix(".jsonl.idx").is_file()


def test_stateless_sampler_multi_files_with_scales(cuts_files: Tuple[Path]):
    index_path = cuts_files[0].parent / "cuts.idx"

    # Run this test 10 times to ensure cutset A is always more likely than B due to scale usage.
    # Remember PoveySampler is non-deterministic.
    for _ in range(10):

        sampler = StatelessSampler(
            [
                # cutset A is 100x more likely than cutset B given their sizes are equal
                (cuts_files[0], 100),
                (cuts_files[1], 1),
            ],
            index_path=index_path,
            max_cuts=2,
            base_seed=0,
        )

        cuts_A_seen = 0
        cuts_B_seen = 0
        for idx, batch in enumerate(sampler):
            for cut in batch:
                if int(cut.id.split("_")[0].split("-")[-1]) < 10:
                    cuts_A_seen += 1
                else:
                    cuts_B_seen += 1

            if idx == 4:
                break  # the sampler is infinite

        assert cuts_A_seen > cuts_B_seen  # oversampling works

        assert sampler.diagnostics.total_batches == 5
        assert sampler.diagnostics.total_cuts == 10

        assert index_path.is_file()
        assert cuts_files[0].with_suffix(".jsonl.idx").is_file()
        assert cuts_files[1].with_suffix(".jsonl.idx").is_file()


class _DummyDataset(torch.utils.data.Dataset):
    def __getitem__(self, cuts: CutSet) -> CutSet:
        return cuts


@pytest.mark.parametrize("num_workers", [0, 2])
def test_stateless_sampler_in_dataloader(cuts_files: Tuple[Path], num_workers: int):
    index_path = cuts_files[0].parent / "cuts.idx"
    sampler = StatelessSampler(
        cuts_files, index_path=index_path, max_cuts=2, base_seed=0
    )
    dloader = torch.utils.data.DataLoader(
        # Note: map-dataset keeps the sampler in the main process,
        #       so each worker is going to see worker_id=0,
        #       but can still be random due to seed sampling with system TRNG
        _DummyDataset(),
        sampler=sampler,
        batch_size=None,
        num_workers=num_workers,
    )

    cut_id_counts = Counter()

    for idx, batch in enumerate(dloader):
        assert len(batch) == 2
        for cut in batch:
            # The cut IDs will be in range 0-19,
            # with first file having 0-9 and second 10-19.
            assert re.match(r"dummy-mono-cut-00[01]\d.*", cut.id) is not None, cut.id
            cut_id_counts[cut.id.split("_")[0]] += 1
        if idx == 49:
            break  # the sampler is infinite

    # With small data and not enough iterations there will always be some duplication:
    # we leverage this property to test this class.
    # counts is a list of tuples like [("id1", 10), ("id2", 8), ("id3", 7), ...]
    counts = cut_id_counts.most_common()
    assert counts[0][1] - counts[-1][1] > 1, counts
    assert sampler.diagnostics.current_epoch == 0  # epoch is never incremented


@pytest.mark.parametrize("num_workers", [0, 2])
def test_stateless_sampler_in_dataloader_with_iterable_dataset(
    cuts_files: Tuple[Path], num_workers: int
):
    index_path = cuts_files[0].parent / "cuts.idx"
    sampler = StatelessSampler(
        cuts_files, index_path=index_path, max_cuts=2, base_seed=0
    )
    dloader = torch.utils.data.DataLoader(
        # Note: iterable dataset will move the sampler into worker subprocess,
        #       which will result in correctly assigning the worker_id
        IterableDatasetWrapper(dataset=_DummyDataset(), sampler=sampler),
        batch_size=None,
        num_workers=num_workers,
    )

    cut_id_counts = Counter()

    for idx, batch in enumerate(dloader):
        assert len(batch) == 2
        for cut in batch:
            # The cut IDs will be in range 0-19,
            # with first file having 0-9 and second 10-19.
            assert re.match(r"dummy-mono-cut-00[01]\d.*", cut.id) is not None, cut.id
            cut_id_counts[cut.id.split("_")[0]] += 1
        if idx == 49:
            break  # the sampler is infinite

    # With small data and not enough iterations there will always be some duplication:
    # we leverage this property to test this class.
    # counts is a list of tuples like [("id1", 10), ("id2", 8), ("id3", 7), ...]
    counts = cut_id_counts.most_common()
    assert counts[0][1] - counts[-1][1] > 1, counts
    assert sampler.diagnostics.current_epoch == 0  # epoch is never incremented


def test_stateless_sampler_bucketing(cuts_files: Tuple[Path]):
    index_path = cuts_files[0].parent / "cuts.idx"
    sampler = StatelessSampler(
        cuts_files,
        index_path=index_path,
        duration_bins=[1.5],
        max_duration=4,
        base_seed=0,
    )

    for idx, batch in enumerate(sampler):
        assert sum(c.duration for c in batch) <= 4
        num_cuts = len(batch)
        for cut in batch:
            if num_cuts == 2:
                assert cut.duration == 2
            else:  # == 4
                assert cut.duration == 1
        if idx == 4:
            break  # the sampler is infinite

    assert sampler.diagnostics.total_batches == 5
    assert 10 <= sampler.diagnostics.total_cuts <= 20


def test_stateless_sampler_requires_uncompressed_manifest():
    with pytest.raises(
        AssertionError, match="^We only support uncompressed .jsonl files.+"
    ):
        with as_lazy(
            DummyManifest(CutSet, begin_id=0, end_id=10), suffix=".jsonl.gz"
        ) as cuts:
            path = Path(cuts.data.path)
            index_path = path.with_suffix(".idx")
            # Call below will raise due to gzip compression
            sampler = StatelessSampler(
                path, index_path=index_path, max_cuts=2, base_seed=0
            )


def test_stateless_sampler_base_seed_is_deterministic(cuts_files: Tuple[Path]):
    path = cuts_files[0]
    index_path = cuts_files[0].parent / "cuts.idx"
    sampler = StatelessSampler(path, index_path=index_path, max_cuts=2, base_seed=0)

    b1 = []
    for idx, b in enumerate(sampler):
        b1.append(b)
        if idx == 10:
            break  # the sampler is infinite
    b2 = []
    for idx, b in enumerate(sampler):
        b2.append(b)
        if idx == 10:
            break  # the sampler is infinite

    assert b1 == b2


def test_stateless_sampler_map(cuts_files: Tuple[Path]):
    path = cuts_files[0]
    index_path = cuts_files[0].parent / "cuts.idx"
    transform = CutConcatenate(gap=0.0, duration_factor=5.0)  # will glue 5 cuts into 1

    sampler = StatelessSampler(path, index_path, max_duration=5.0, base_seed=0)
    sampler.map(transform)

    b = next(iter(sampler))
    assert len(b) == 1
    assert b[0].duration == 5.0
