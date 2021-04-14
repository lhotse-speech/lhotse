import tempfile

import pytest
import torch

from lhotse import CutSet
from lhotse.dataset import GlobalMVN, RandomizedSmoothing, SpecAugment
from lhotse.dataset.collation import collate_features


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


def test_global_mvn_from_cuts():
    cuts = CutSet.from_json('test/fixtures/ljspeech/cuts.json')
    stats1 = GlobalMVN.from_cuts(cuts)
    stats2 = GlobalMVN.from_cuts(cuts, max_cuts=1)
    assert isinstance(stats1, GlobalMVN)
    assert isinstance(stats2, GlobalMVN)


def test_specaugment_single():
    cuts = CutSet.from_json('test/fixtures/ljspeech/cuts.json')
    feats = torch.from_numpy(cuts[0].load_features())
    tfnm = SpecAugment(p=1.0, time_warp_factor=10)
    augmented = tfnm(feats)
    assert (feats != augmented).any()


@pytest.mark.parametrize('num_feature_masks', [0, 1, 2])
@pytest.mark.parametrize('num_frame_masks', [0, 1, 2])
def test_specaugment_batch(num_feature_masks, num_frame_masks):
    cuts = CutSet.from_json('test/fixtures/ljspeech/cuts.json')
    feats, feat_lens = collate_features(cuts)
    tfnm = SpecAugment(
        p=1.0,
        time_warp_factor=10,
        features_mask_size=5,
        frames_mask_size=20,
        num_feature_masks=num_feature_masks,
        num_frame_masks=num_frame_masks
    )
    augmented = tfnm(feats)
    assert (feats != augmented).any()


@pytest.mark.parametrize('sample_sigma', [True, False])
def test_randomized_smoothing(sample_sigma):
    audio = torch.zeros(64, 4000, dtype=torch.float32)
    tfnm = RandomizedSmoothing(sigma=0.1, sample_sigma=sample_sigma, p=0.8)
    audio_aug = tfnm(audio)
    # Shapes are the same
    assert audio.shape == audio_aug.shape
    # All samples are different than the input audio
    assert (audio != audio_aug).any()
    # Different batch samples receive different augmentation:
    # we sum along the time axis and compare the summed values;
    # if all examples got the same augmentation,
    # there would have been just one unique value.
    assert len(set(audio_aug.sum(dim=1).tolist())) > 1


def test_randomized_smoothing_p1():
    audio = torch.zeros(64, 4000, dtype=torch.float32)
    tfnm = RandomizedSmoothing(sigma=0.1, p=1.0)
    audio_aug = tfnm(audio)
    # Shapes are the same
    assert audio.shape == audio_aug.shape
    # All samples are different than the input audio
    assert (audio != audio_aug).all()
    # Different batch samples receive different augmentation
    assert (audio_aug[0] != audio_aug[1]).all()


def test_randomized_smoothing_p0():
    audio = torch.zeros(64, 4000, dtype=torch.float32)
    tfnm = RandomizedSmoothing(sigma=0.1, p=0.0)
    audio_aug = tfnm(audio)
    # Shapes are the same
    assert audio.shape == audio_aug.shape
    # Audio is unaffacted
    assert (audio == audio_aug).all()
    # Audio is unaffacted across batches
    assert (audio_aug[0] == audio_aug[1]).all()


def test_randomized_smoothing_schedule():
    audio = torch.zeros(16, 16000, dtype=torch.float32)
    tfnm = RandomizedSmoothing(
        sigma=[
            (0, 0.01),
            (100, 0.5)
        ],
        p=0.8
    )
    audio_aug = tfnm(audio)
    # Shapes are the same
    assert audio.shape == audio_aug.shape
    # All samples are different than the input audio
    assert (audio != audio_aug).any()
    # Different batch samples receive different augmentation:
    # we sum along the time axis and compare the summed values;
    # if all examples got the same augmentation,
    # there would have been just one unique value.
    assert len(set(audio_aug.sum(dim=1).tolist())) > 1

    tfnm.step = 1000
    audio_aug2 = tfnm(audio)
    # The schedule kicked in and the abs magnitudes should be larger.
    assert audio_aug2.abs().sum() > audio_aug.abs().sum()
