from typing import List
from uuid import uuid4

import numpy as np
import pytest
import torch

from lhotse import CutSet
from lhotse.dataset import DynamicCutSampler, IterableDatasetWrapper
from lhotse.testing.dummies import DummyManifest
from lhotse.testing.random import deterministic_rng


class DummyDataset(torch.utils.data.Dataset):
    def __getitem__(self, item):
        return item


def mark(val: int):
    def _inner(cut):
        cut.source = val
        return cut

    return _inner


def random_id(*args):
    return str(uuid4())


def assert_sources_are(cuts: CutSet, expected: List[int]):
    actual = [c.source for c in cuts]
    assert actual == expected


@pytest.mark.parametrize("weight_type", [list, np.array, torch.tensor])
def test_mux_with_controllable_weights(deterministic_rng, weight_type):
    """The sampler and the worker are both in the main process."""

    # 3 infinite iterables
    cuts1 = DummyManifest(CutSet, begin_id=0, end_id=3).map(mark(0)).repeat()
    cuts2 = DummyManifest(CutSet, begin_id=10, end_id=13).map(mark(1)).repeat()
    cuts3 = DummyManifest(CutSet, begin_id=100, end_id=103).map(mark(2)).repeat()

    weights = weight_type([1, 0, 0])
    muxd = CutSet.mux(cuts1, cuts2, cuts3, weights=weights)

    dloader = torch.utils.data.DataLoader(
        dataset=DummyDataset(),
        sampler=DynamicCutSampler(muxd, max_cuts=2),
        batch_size=None,
        num_workers=0,
    )

    dloader = iter(dloader)
    b = next(dloader)
    assert_sources_are(b, [0, 0])

    weights[0] = 0
    weights[1] = 1
    b = next(dloader)
    assert_sources_are(b, [1, 1])

    weights[1] = 0
    weights[2] = 1
    b = next(dloader)
    assert_sources_are(b, [2, 2])


def test_mux_with_controllable_weights_subprocess_worker(deterministic_rng):
    """
    The sampler is in the main process but the worker is in a sub-process.

    In general expect a latency of ``prefetch_factor * num_workers`` in the propagation
    of weights between the main process and the dataloading subprocesses.
    """

    # 3 infinite iterables
    cuts1 = DummyManifest(CutSet, begin_id=0, end_id=3).map(mark(0)).repeat()
    cuts2 = DummyManifest(CutSet, begin_id=10, end_id=13).map(mark(1)).repeat()
    cuts3 = DummyManifest(CutSet, begin_id=100, end_id=103).map(mark(2)).repeat()

    weights = [1, 0, 0]
    muxd = CutSet.mux(cuts1, cuts2, cuts3, weights=weights)

    dloader = torch.utils.data.DataLoader(
        dataset=DummyDataset(),
        sampler=DynamicCutSampler(muxd, max_cuts=2),
        batch_size=None,
        num_workers=1,
        prefetch_factor=1,
    )

    dloader = iter(dloader)
    b = next(dloader)
    assert_sources_are(b, [0, 0])

    weights[0] = 0
    weights[1] = 1
    b = next(dloader)
    assert_sources_are(
        b, [0, 0]
    )  # prefetch_factor causes one batch with previous weights to be retained
    b = next(dloader)
    assert_sources_are(b, [1, 1])

    weights[1] = 0
    weights[2] = 1
    b = next(dloader)
    assert_sources_are(
        b, [1, 1]
    )  # prefetch_factor causes one batch with previous weights to be retained
    b = next(dloader)
    assert_sources_are(b, [2, 2])


@pytest.mark.xfail(reason="This test is flaky...")
def test_mux_with_controllable_weights_subprocess_sampler_shared_memory(
    deterministic_rng,
):
    """
    The sampler is placed in the dataloading subprocess.

    Note: we are using PyTorch shared memory to share the weight tensor across processes.

    In general expect a latency of ``prefetch_factor * num_workers`` in the propagation
    of weights between the main process and the dataloading subprocesses.
    """

    # 3 infinite iterables
    cuts1 = DummyManifest(CutSet, begin_id=0, end_id=3).map(mark(0)).repeat()
    cuts2 = DummyManifest(CutSet, begin_id=10, end_id=13).map(mark(1)).repeat()
    cuts3 = DummyManifest(CutSet, begin_id=100, end_id=103).map(mark(2)).repeat()

    weights = torch.tensor([1, 0, 0]).share_memory_()
    assert weights.is_shared()
    muxd = CutSet.mux(cuts1, cuts2, cuts3, weights=weights)

    dloader = torch.utils.data.DataLoader(
        dataset=IterableDatasetWrapper(
            dataset=DummyDataset(), sampler=DynamicCutSampler(muxd, max_cuts=2)
        ),
        batch_size=None,
        num_workers=1,
        prefetch_factor=1,
    )

    dloader = iter(dloader)
    b = next(dloader)
    assert_sources_are(b, [0, 0])

    weights[:] = torch.tensor([0, 1, 0])  # atomic update
    b = next(dloader)
    assert_sources_are(b, [1, 1])

    weights[:] = torch.tensor([0, 0, 1])  # atomic update
    b = next(dloader)
    assert_sources_are(b, [2, 2])


@pytest.mark.xfail(reason="This test is flaky...")
def test_infinite_mux_with_controllable_weights_subprocess_sampler_shared_memory(
    deterministic_rng,
):
    """
    The sampler is placed in the dataloading subprocess.

    Note: we are using PyTorch shared memory to share the weight tensor across processes.

    In general expect a latency of ``prefetch_factor * num_workers`` in the propagation
    of weights between the main process and the dataloading subprocesses.
    """

    # 3 infinite iterables
    cuts1 = DummyManifest(CutSet, begin_id=0, end_id=3).map(mark(0))
    cuts2 = DummyManifest(CutSet, begin_id=10, end_id=13).map(mark(1))
    cuts3 = DummyManifest(CutSet, begin_id=100, end_id=103).map(mark(2))

    weights = torch.tensor([1, 0, 0]).share_memory_()
    assert weights.is_shared()
    # randomize_id is required because infinite_mux may sample the same cut in a mini batch
    muxd = CutSet.infinite_mux(cuts1, cuts2, cuts3, weights=weights).modify_ids(
        random_id
    )

    dloader = torch.utils.data.DataLoader(
        dataset=IterableDatasetWrapper(
            dataset=DummyDataset(), sampler=DynamicCutSampler(muxd, max_cuts=2)
        ),
        batch_size=None,
        num_workers=1,
        prefetch_factor=1,
    )

    dloader = iter(dloader)
    b = next(dloader)
    assert_sources_are(b, [0, 0])

    # Note the latency for several batches. The reason is the following:
    #   infinite_mux() samples streams with replacement, and at the beginning of the test is samples
    #   3x stream #0, which has 3 items each with equal probability.
    #   It will only start returning items from stream #1 once one of the previous streams is exhausted.
    weights[:] = torch.tensor([0, 1, 0])  # atomic update
    b = next(dloader)
    assert_sources_are(b, [0, 0])
    b = next(dloader)
    assert_sources_are(b, [0, 0])
    b = next(dloader)
    assert_sources_are(b, [1, 1])

    # The latency strikes again as now we have both streams #0 and #1 open,
    # but they have zero weight. It means they will be uniformly sampled until
    # one of them is exhausted and a new stream #2 is opened.
    weights[:] = torch.tensor([0, 0, 1])  # atomic update
    b = next(dloader)
    assert_sources_are(b, [0, 0])
    b = next(dloader)
    assert_sources_are(b, [1, 2])
    b = next(dloader)
    assert_sources_are(b, [2, 2])
