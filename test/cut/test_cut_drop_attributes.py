import pytest

from lhotse import CutSet
from lhotse.cut import PaddingCut
from lhotse.testing.dummies import dummy_cut, dummy_supervision

parametrize_on_cut_types = pytest.mark.parametrize(
    "cut",
    [
        # MonoCut
        dummy_cut(0, supervisions=[dummy_supervision(0)]),
        # PaddingCut
        PaddingCut(
            "pad",
            duration=1.0,
            sampling_rate=16000,
            feat_value=-100,
            num_frames=100,
            frame_shift=0.01,
            num_features=80,
            num_samples=16000,
        ),
        # MixedCut
        dummy_cut(0, supervisions=[dummy_supervision(0)]).mix(
            dummy_cut(1, supervisions=[dummy_supervision(1)]),
            offset_other_by=0.5,
            snr=10,
        ),
    ],
)


@parametrize_on_cut_types
def test_drop_features(cut):
    assert cut.has_features
    cut_drop = cut.drop_features()
    assert cut.has_features
    assert not cut_drop.has_features


@parametrize_on_cut_types
def test_drop_recording(cut):
    assert cut.has_recording
    cut_drop = cut.drop_recording()
    assert cut.has_recording
    assert not cut_drop.has_recording


@parametrize_on_cut_types
def test_drop_supervisions(cut):
    assert len(cut.supervisions) > 0 or isinstance(cut, PaddingCut)
    cut_drop = cut.drop_supervisions()
    assert len(cut.supervisions) > 0 or isinstance(cut, PaddingCut)
    assert len(cut_drop.supervisions) == 0


@pytest.fixture()
def cutset():
    return CutSet.from_cuts(
        [
            # MonoCut
            dummy_cut(0, supervisions=[dummy_supervision(0)]),
            # PaddingCut
            PaddingCut(
                "pad",
                duration=1.0,
                sampling_rate=16000,
                feat_value=-100,
                num_frames=100,
                frame_shift=0.01,
                num_features=80,
                num_samples=16000,
            ),
            # MixedCut
            dummy_cut(0, supervisions=[dummy_supervision(0)]).mix(
                dummy_cut(1, supervisions=[dummy_supervision(1)]),
                offset_other_by=0.5,
                snr=10,
            ),
        ]
    )


def test_drop_features_cutset(cutset):
    assert any(cut.has_features for cut in cutset)
    cutset_drop = cutset.drop_features()
    assert any(cut.has_features for cut in cutset)
    assert all(not cut.has_features for cut in cutset_drop)


def test_drop_recordings_cutset(cutset):
    assert any(cut.has_recording for cut in cutset)
    cutset_drop = cutset.drop_recordings()
    assert any(cut.has_recording for cut in cutset)
    assert all(not cut.has_recording for cut in cutset_drop)


def test_drop_supervisions_cutset(cutset):
    assert any(len(cut.supervisions) > 0 for cut in cutset)
    cutset_drop = cutset.drop_supervisions()
    assert any(len(cut.supervisions) > 0 for cut in cutset)
    assert all(len(cut.supervisions) == 0 for cut in cutset_drop)
