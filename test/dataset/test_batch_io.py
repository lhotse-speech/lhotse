from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial

import pytest

from lhotse import CutSet, Fbank
from lhotse.dataset import AudioSamples, OnTheFlyFeatures, PrecomputedFeatures


@pytest.fixture
def libri_cut_set():
    cuts = CutSet.from_json("test/fixtures/libri/cuts.json")
    return CutSet.from_cuts(
        [
            cuts[0],
            cuts[0].with_id("copy-1"),
            cuts[0].with_id("copy-2"),
            cuts[0].append(cuts[0]),
        ]
    )


@pytest.mark.parametrize(
    "batchio", [AudioSamples, PrecomputedFeatures, partial(OnTheFlyFeatures, Fbank())]
)
@pytest.mark.parametrize("num_workers", [0, 1, 2])
@pytest.mark.parametrize("executor_type", [ThreadPoolExecutor, ProcessPoolExecutor])
def test_batch_io(libri_cut_set, batchio, num_workers, executor_type):
    # does not fail / hang / etc.
    read_fn = batchio(num_workers=num_workers, executor_type=executor_type)
    read_fn(libri_cut_set)
