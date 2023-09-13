import tempfile

import pytest
import torch

from lhotse import CutSet
from lhotse.dataset import GlobalMVN, RandomizedSmoothing, SpecAugment
from lhotse.dataset.collation import collate_features
from lhotse.dataset.signal_transforms import DereverbWPE
from lhotse.testing.random import deterministic_rng
from lhotse.utils import is_module_available


@pytest.fixture
def global_mvn():
    cuts = CutSet.from_json("test/fixtures/ljspeech/cuts.json")
    return GlobalMVN.from_cuts(cuts)


def test_global_mvn_initialization_and_stats_saving(global_mvn):
    with tempfile.NamedTemporaryFile() as tf:
        global_mvn.to_file(tf.name)
        global_mvn2 = GlobalMVN.from_file(tf.name)

    for key_item_1, key_item_2 in zip(
        global_mvn.state_dict().items(), global_mvn2.state_dict().items()
    ):
        assert torch.equal(key_item_1[1], key_item_2[1])


@pytest.mark.parametrize("in_tensor", [torch.ones(10, 40), torch.ones(2, 10, 40)])
def test_global_mvn_shapes(global_mvn, in_tensor):
    assert global_mvn(in_tensor).shape == in_tensor.shape
    assert global_mvn.inverse(in_tensor).shape == in_tensor.shape


@pytest.mark.parametrize("in_tensor", [torch.ones(10, 40), torch.ones(2, 10, 40)])
def test_global_mvn_inverse(global_mvn, in_tensor):
    out_tensor = global_mvn(in_tensor)
    assert torch.allclose(in_tensor, global_mvn.inverse(out_tensor))


def test_global_mvn_from_cuts():
    cuts = CutSet.from_json("test/fixtures/ljspeech/cuts.json")
    stats1 = GlobalMVN.from_cuts(cuts)
    stats2 = GlobalMVN.from_cuts(cuts, max_cuts=1)
    assert isinstance(stats1, GlobalMVN)
    assert isinstance(stats2, GlobalMVN)


def test_specaugment_2d_input_raises_error():
    cuts = CutSet.from_json("test/fixtures/ljspeech/cuts.json")
    feats = torch.from_numpy(cuts[0].load_features())
    tfnm = SpecAugment(p=1.0, time_warp_factor=10)
    with pytest.raises(AssertionError):
        augmented = tfnm(feats)
        assert (feats != augmented).any()


@pytest.mark.parametrize("num_feature_masks", [0, 1, 2])
@pytest.mark.parametrize("num_frame_masks", [1, 2, 3])
def test_specaugment_3d_input_works(
    deterministic_rng, num_feature_masks, num_frame_masks
):
    cuts = CutSet.from_json("test/fixtures/ljspeech/cuts.json")
    feats, feat_lens = collate_features(cuts)
    tfnm = SpecAugment(
        p=1.0,
        time_warp_factor=10,
        features_mask_size=5,
        frames_mask_size=20,
        num_feature_masks=num_feature_masks,
        num_frame_masks=num_frame_masks,
    )
    augmented = tfnm(feats)
    assert (feats != augmented).any()


def test_specaugment_state_dict():
    # all values default
    config = dict(
        time_warp_factor=80,
        num_feature_masks=1,
        features_mask_size=13,
        num_frame_masks=1,
        frames_mask_size=70,
        max_frames_mask_fraction=0.2,
        p=0.5,
    )
    specaug = SpecAugment(**config)
    state_dict = specaug.state_dict()

    for key, value in config.items():
        assert state_dict[key] == value


def test_specaugment_load_state_dict():
    torch.manual_seed(0)
    # all values non-default
    config = dict(
        time_warp_factor=85,
        num_feature_masks=2,
        features_mask_size=12,
        num_frame_masks=2,
        frames_mask_size=71,
        max_frames_mask_fraction=0.25,
        p=0.6,
    )
    specaug = SpecAugment()
    specaug.load_state_dict(config)

    for key, value in config.items():
        assert getattr(specaug, key) == value


@pytest.mark.parametrize("sample_sigma", [True, False])
def test_randomized_smoothing(deterministic_rng, sample_sigma):
    torch.manual_seed(0)
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


def test_randomized_smoothing_p1(deterministic_rng):
    audio = torch.zeros(64, 4000, dtype=torch.float32)
    tfnm = RandomizedSmoothing(sigma=0.1, p=1.0)
    audio_aug = tfnm(audio)
    # Shapes are the same
    assert audio.shape == audio_aug.shape
    # Some (most) samples are different than the input audio
    assert (audio != audio_aug).any()
    # Different batch samples receive different augmentation
    assert (audio_aug[0] != audio_aug[1]).any()


def test_randomized_smoothing_p0(deterministic_rng):
    audio = torch.zeros(64, 4000, dtype=torch.float32)
    tfnm = RandomizedSmoothing(sigma=0.1, p=0.0)
    audio_aug = tfnm(audio)
    # Shapes are the same
    assert audio.shape == audio_aug.shape
    # Audio is unaffacted
    assert (audio == audio_aug).all()
    # Audio is unaffacted across batches
    assert (audio_aug[0] == audio_aug[1]).all()


def test_randomized_smoothing_schedule(deterministic_rng):
    audio = torch.zeros(16, 16000, dtype=torch.float32)
    tfnm = RandomizedSmoothing(sigma=[(0, 0.01), (100, 0.5)], p=0.8)
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


@pytest.mark.skipif(
    not is_module_available("nara_wpe"), reason="Requires nara_wpe to be installed."
)
def test_wpe_single_channel(deterministic_rng):
    B, T = 16, 32000
    audio = torch.randn(B, T, dtype=torch.float32)
    tfnm = DereverbWPE()
    audio_aug = tfnm(audio)
    # Shapes are the same
    assert audio.shape == audio_aug.shape
    # Some samples are different than the input audio
    assert (audio != audio_aug).any()


@pytest.mark.skipif(
    not is_module_available("nara_wpe"), reason="Requires nara_wpe to be installed."
)
def test_wpe_multi_channel(deterministic_rng):
    B, D, T = 16, 2, 32000
    audio = torch.randn(B, D, T, dtype=torch.float32)
    tfnm = DereverbWPE()
    audio_aug = tfnm(audio)
    # Shapes are the same
    assert audio.shape == audio_aug.shape
    # Some samples are different than the input audio
    assert (audio != audio_aug).any()
