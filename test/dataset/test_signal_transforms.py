import tempfile

import pytest
import torch

from lhotse import CutSet
from lhotse.dataset.signal_transforms import GlobalMVN


@pytest.fixture
def global_mvn():
    cuts = CutSet.from_json('test/fixtures/ljspeech/cuts.json')
    return GlobalMVN.from_cuts(cuts)


def test_global_mvn_initialization_and_stats_saving(global_mvn):
    with tempfile.NamedTemporaryFile() as tf:
        global_mvn.to_file(tf.name)
        global_mvn2 = GlobalMVN.from_file(tf.name)

    for key_item_1, key_item_2 in zip(
            global_mvn.state_dict().items(),
            global_mvn2.state_dict().items()
    ):
        assert torch.equal(key_item_1[1], key_item_2[1])


@pytest.mark.parametrize(
    "in_tensor", [torch.ones(10, 40), torch.ones(2, 10, 40)]
)
def test_global_mvn_shapes(global_mvn, in_tensor):
    assert global_mvn(in_tensor).shape == in_tensor.shape
    assert global_mvn.inverse(in_tensor).shape == in_tensor.shape


@pytest.mark.parametrize(
    "in_tensor", [torch.ones(10, 40), torch.ones(2, 10, 40)]
)
def test_global_mvn_inverse(global_mvn, in_tensor):
    out_tensor = global_mvn(in_tensor)
    assert torch.allclose(in_tensor, global_mvn.inverse(out_tensor))
