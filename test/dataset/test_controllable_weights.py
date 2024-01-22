import torch

from lhotse import CutSet
from lhotse.dataset import DynamicCutSampler, IterableDatasetWrapper
from lhotse.testing.dummies import DummyManifest


class DummyDataset(torch.utils.data.Dataset):
    def __getitem__(self, item):
        return item


def test_mux_with_controllable_weights():
    def mark(val: int):
        def _inner(cut):
            cut.source = val
            return cut

        return _inner

    # 3 infinite iterables
    cuts1 = DummyManifest(CutSet, begin_id=0, end_id=3).map(mark(0)).repeat()
    cuts2 = DummyManifest(CutSet, begin_id=10, end_id=13).map(mark(1)).repeat()
    cuts3 = DummyManifest(CutSet, begin_id=100, end_id=103).map(mark(2)).repeat()

    def assert_sources_are(cuts: CutSet, expected: list[int]):
        actual = [c.source for c in cuts]
        assert actual == expected

    # TODO: initialize weights
    weights = [1, 0, 0]

    muxd = CutSet.mux(cuts1, cuts2, cuts3, weights=weights)

    sampler = DynamicCutSampler(muxd, max_cuts=2)

    # locate the sampler in a sub-process
    dloader = torch.utils.data.DataLoader(
        dataset=DummyDataset(),
        sampler=sampler,
        batch_size=None,
        num_workers=0,
    )

    dloader = iter(dloader)
    b = next(dloader)
    assert_sources_are(b, [0, 0])

    # TODO: set the weight
    weights[0] = 0
    weights[1] = 1
    b = next(dloader)
    assert_sources_are(b, [1, 1])

    # TODO: set the weight
    weights[1] = 0
    weights[2] = 1
    b = next(dloader)
    assert_sources_are(b, [2, 2])


def test_mux_with_controllable_weights_multiprocess():
    return
    # # 3 infinite iterables
    # cuts1 = DummyManifest(CutSet, begin_id=0, end_id=3).repeat()
    # cuts2 = DummyManifest(CutSet, begin_id=10, end_id=13).repeat()
    # cuts3 = DummyManifest(CutSet, begin_id=100, end_id=103).repeat()
    #
    # weights = [1, 1, 1]
    #
    # muxd = CutSet.mux(cuts1, cuts2, cuts3, weights=weights)
    #
    # sampler = DynamicCutSampler(muxd, max_cuts=2)
    #
    # # locate the sampler in a sub-process
    # dloader = torch.utils.data.DataLoader(
    #     dataset=IterableDatasetWrapper(dataset=DummyDataset(), sampler=sampler),
    #     batch_size=None,
    #     num_workers=1,
    # )
