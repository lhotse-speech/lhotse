import pytest
from torch.utils.data import DataLoader

from lhotse.cut import CutSet
from lhotse.dataset.sampling import SimpleCutSampler
from lhotse.dataset.surt import K2SurtDataset


@pytest.fixture
def cut_set():
    return CutSet.from_shar(in_dir="test/fixtures/lsmix")


@pytest.mark.parametrize("num_workers", [0, 1])
@pytest.mark.parametrize("return_sources", [True, False])
def test_surt_iterable_dataset(cut_set, num_workers, return_sources):
    dataset = K2SurtDataset(return_sources=return_sources, return_cuts=True)
    sampler = SimpleCutSampler(cut_set, shuffle=False, max_cuts=10000)
    # Note: "batch_size=None" disables the automatic batching mechanism,
    #       which is required when Dataset takes care of the collation itself.
    dloader = DataLoader(
        dataset, batch_size=None, sampler=sampler, num_workers=num_workers
    )
    batch = next(iter(dloader))
    assert batch["inputs"].shape == (2, 2238, 80)
    assert batch["input_lens"].tolist() == [2238, 985]

    assert len(batch["supervisions"][1]) == 2
    assert len(batch["text"][1]) == 2
    assert batch["text"][1] == [
        "BY THIS MANOEUVRE WE DON'T LET ANYBODY IN THE CAR AND WE TRY AND KEEP THEM CLEAR OF THE CAR SHORT OF SHOOTING THEM THAT IS CARRIED NO OTHER MESSAGE",
        "THE AMERICAN INTERPOSED BRUSQUELY BETWEEN PAROXYSMS AND THEY CAUGHT HIM AT IT EH",
    ]
    if return_sources:
        assert len(batch["source_feats"]) == 2
        assert all(
            len(batch["source_feats"][i]) == len(batch["cuts"][i].supervisions)
            for i in range(2)
        )
