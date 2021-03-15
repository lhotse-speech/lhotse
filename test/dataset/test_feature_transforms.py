import tempfile
import torch

from lhotse import CutSet
from lhotse.dataset.feature_transforms import GlobalMVN


def test_global_mvn_initialization_and_stats_saving():
    cuts = CutSet.from_json('test/fixtures/ljspeech/cuts.json')
    tf = tempfile.NamedTemporaryFile()

    global_mvn = GlobalMVN.from_cuts(cuts)
    global_mvn.to_file(tf.name)
    global_mvn2 = GlobalMVN.from_file(tf.name)
    tf.close()

    for key_item_1, key_item_2 in zip(global_mvn.state_dict().items(), global_mvn2.state_dict().items()):
        assert torch.equal(key_item_1[1], key_item_2[1])


