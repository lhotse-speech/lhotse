import numpy as np
import pytest

from lhotse.cut import PaddingCut, Cut
from lhotse.features import Features

PADDING_ENERGY = 1e-8
PADDING_LOG_ENERGY = -18.420680743952367  # log(1e-8)


@pytest.fixture
def padding_cut():
    return PaddingCut(
        id='padding-1',
        duration=10.0,
        num_frames=1000,
        num_features=23,
        sampling_rate=16000,
        use_log_energy=True
    )


@pytest.mark.parametrize(
    ['use_log_energy', 'expected_value'],
    [
        (True, PADDING_LOG_ENERGY),
        (False, PADDING_ENERGY)
    ]
)
def test_load_features_log(padding_cut, use_log_energy, expected_value):
    padding_cut.use_log_energy = use_log_energy
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


@pytest.mark.parametrize(
    ['offset', 'duration', 'expected_duration', 'expected_num_frames', 'expected_num_samples'],
    [
        (0.0, None, 10.0, 1000, 160000),
        (0.0, 5.0, 5.0, 500, 80000),
        (5.0, None, 5.0, 500, 80000),
        (5.0, 5.0, 5.0, 500, 80000),
        (5.0, 2.0, 2.0, 200, 32000),
    ]
)
def test_truncate(padding_cut, offset, duration, expected_duration, expected_num_frames, expected_num_samples):
    cut = padding_cut.truncate(offset=offset, duration=duration, preserve_id=True)
    # Invariants
    assert cut.frame_shift == padding_cut.frame_shift
    assert cut.num_features == padding_cut.num_features
    assert cut.sampling_rate == padding_cut.sampling_rate
    assert cut.use_log_energy == padding_cut.use_log_energy
    assert cut.id == padding_cut.id
    # Variants
    assert cut.duration == expected_duration
    assert cut.num_frames == expected_num_frames
    assert cut.load_features().shape == (expected_num_frames, 23)
    assert cut.load_audio().shape == (1, expected_num_samples)


@pytest.fixture
def libri_cut():
    return Cut(
        duration=16.04,
        features=Features(
            channel_id=0,
            duration=16.04,
            num_features=23,
            num_frames=1604,
            recording_id='recording-1',
            sampling_rate=16000,
            start=0.0,
            storage_path='test/fixtures/libri/storage/dc2e0952-f2f8-423c-9b8c-f5481652ee1d.llc',
            storage_type='lilcom',
            type='fbank',
        ),
        id='849e13d8-61a2-4d09-a542-dac1aee1b544',
        start=0.0,
        supervisions=[],
    )


def test_overlay_in_the_middle(libri_cut, padding_cut):
    mixed = libri_cut.overlay(padding_cut)

    # Invariants
    assert mixed.duration == 16.04
    assert mixed.num_features == 23
    assert mixed.num_frames == 1604

    # Check that the actual feature shapes and values did not change
    pre_mixed_feats = libri_cut.load_features()
    mixed_feats = mixed.load_features()
    assert mixed_feats.shape == pre_mixed_feats.shape
    np.testing.assert_allclose(pre_mixed_feats, mixed_feats, rtol=1e-2)


def test_overlay_pad_right(libri_cut, padding_cut):
    mixed = libri_cut.overlay(padding_cut, offset_other_by=10.0)

    # Invariants
    assert mixed.num_features == 23

    # Variants
    assert mixed.duration == 20.0
    assert mixed.num_frames == 2000

    mixed_feats = mixed.load_features()
    assert mixed_feats.shape == (2000, 23)
    np.testing.assert_equal(mixed_feats[1604:, :], PADDING_LOG_ENERGY)  # Only padding after 16.04s
    np.testing.assert_array_less(PADDING_LOG_ENERGY, mixed_feats[1603, :])  # Padding didn't start before 16.04s

    pre_mixed_feats = libri_cut.load_features()
    np.testing.assert_allclose(pre_mixed_feats, mixed_feats[:1604, :], rtol=1e-2)


def test_overlay_pad_left(libri_cut, padding_cut):
    mixed = padding_cut.overlay(libri_cut, offset_other_by=3.96)

    # Invariants
    assert mixed.num_features == 23

    # Variants
    assert mixed.duration == 20.0
    assert mixed.num_frames == 2000

    mixed_feats = mixed.load_features()
    assert mixed_feats.shape == (2000, 23)
    np.testing.assert_equal(mixed_feats[:396, :], PADDING_LOG_ENERGY)  # Only padding before 3.96s
    np.testing.assert_array_less(PADDING_LOG_ENERGY, mixed_feats[396, :])  # No padding after 3.96s

    pre_mixed_feats = libri_cut.load_features()
    np.testing.assert_allclose(pre_mixed_feats, mixed_feats[396:, :], rtol=1e-2)


@pytest.fixture
def mixed_libri_cut(libri_cut):
    return libri_cut.overlay(libri_cut)


def test_mixed_overlay_in_the_middle(mixed_libri_cut, padding_cut):
    mixed = mixed_libri_cut.overlay(padding_cut)

    # Invariants
    assert mixed.duration == 16.04
    assert mixed.num_features == 23
    assert mixed.num_frames == 1604

    # Check that the actual feature shapes and values did not change
    pre_mixed_feats = mixed_libri_cut.load_features()
    mixed_feats = mixed.load_features()
    assert mixed_feats.shape == pre_mixed_feats.shape
    np.testing.assert_allclose(pre_mixed_feats, mixed_feats, rtol=1e-2)


def test_mixed_overlay_pad_right(mixed_libri_cut, padding_cut):
    mixed = mixed_libri_cut.overlay(padding_cut, offset_other_by=10.0)

    # Invariants
    assert mixed.num_features == 23

    # Variants
    assert mixed.duration == 20.0
    assert mixed.num_frames == 2000

    mixed_feats = mixed.load_features()
    assert mixed_feats.shape == (2000, 23)
    np.testing.assert_equal(mixed_feats[1604:, :], PADDING_LOG_ENERGY)  # Only padding after 16.04s
    np.testing.assert_array_less(PADDING_LOG_ENERGY, mixed_feats[1603, :])  # Padding didn't start before 16.04s

    pre_mixed_feats = mixed_libri_cut.load_features()
    np.testing.assert_allclose(pre_mixed_feats, mixed_feats[:1604, :], rtol=1e-2)


def test_mixed_overlay_pad_left(mixed_libri_cut, padding_cut):
    mixed = padding_cut.overlay(mixed_libri_cut, offset_other_by=3.96)

    # Invariants
    assert mixed.num_features == 23

    # Variants
    assert mixed.duration == 20.0
    assert mixed.num_frames == 2000

    mixed_feats = mixed.load_features()
    assert mixed_feats.shape == (2000, 23)
    np.testing.assert_equal(mixed_feats[:396, :], PADDING_LOG_ENERGY)  # Only padding before 3.96s
    np.testing.assert_array_less(PADDING_LOG_ENERGY, mixed_feats[396, :])  # No padding after 3.96s

    pre_mixed_feats = mixed_libri_cut.load_features()
    np.testing.assert_allclose(pre_mixed_feats, mixed_feats[396:, :], rtol=1e-2)


def test_append(libri_cut, padding_cut):
    appended = libri_cut.append(padding_cut)

    # Invariants
    assert appended.num_features == 23

    # Variants
    assert appended.duration == 26.04
    assert appended.num_frames == 2604

    appended_feats = appended.load_features()
    assert appended_feats.shape == (2604, 23)
    np.testing.assert_equal(appended_feats[1604:, :], PADDING_LOG_ENERGY)  # Only padding after 16.04s
    np.testing.assert_array_less(PADDING_LOG_ENERGY, appended_feats[1603, :])  # Padding didn't start before 16.04s

    original_feats = libri_cut.load_features()
    np.testing.assert_allclose(original_feats, appended_feats[:1604, :], rtol=1e-2)
