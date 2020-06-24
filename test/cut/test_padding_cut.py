import numpy as np
import pytest

from lhotse.cut import PaddingCut


@pytest.fixture
def padding_cut():
    return PaddingCut(
        id='padding-1',
        duration=10.0,
        num_frames=1000,
        num_features=23,
        sampling_rate=16000,
        use_log=True
    )


@pytest.mark.parametrize(
    ['use_log', 'expected_value'],
    [
        (True, -18.420680743952367),  # log(1e-8)
        (False, 1e-8)
    ]
)
def test_load_features_log(padding_cut, use_log, expected_value):
    padding_cut.use_log = use_log
    feats = padding_cut.load_features()
    assert feats.shape[0] == 1000
    assert feats.shape[1] == 23
    np.testing.assert_almost_equal(feats, expected_value)


def test_frame_shift(padding_cut):
    assert padding_cut.frame_shift == 0.01


def test_load_audio(padding_cut):
    samples = padding_cut.load_audio()
    assert samples.shape[0] == 1  # single channel
    assert samples.shape[1] == 160000
    np.testing.assert_equal(samples, 0.0)


def test_overlay(padding_cut):
    mixed_cut = padding_cut.overlay(padding_cut, offset_other_by=5.0)
