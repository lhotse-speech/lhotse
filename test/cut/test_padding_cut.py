from tempfile import NamedTemporaryFile

import numpy as np
import pytest

from lhotse.audio import AudioSource, Recording
from lhotse.cut import CutSet, MonoCut, PaddingCut
from lhotse.features import Features
from lhotse.utils import EPSILON, LOG_EPSILON
from lhotse.utils import nullcontext as does_not_raise

PADDING_ENERGY = EPSILON
PADDING_LOG_ENERGY = LOG_EPSILON


@pytest.fixture
def padding_cut():
    return PaddingCut(
        id="padding-1",
        duration=10.0,
        num_frames=1000,
        num_features=40,
        frame_shift=0.01,
        sampling_rate=16000,
        num_samples=160000,
        feat_value=PADDING_LOG_ENERGY,
    )


@pytest.mark.parametrize("expected_value", [PADDING_LOG_ENERGY, PADDING_ENERGY])
def test_load_features_log(padding_cut, expected_value):
    padding_cut.feat_value = expected_value
    feats = padding_cut.load_features()
    assert feats.shape[0] == 1000
    assert feats.shape[1] == 40
    np.testing.assert_almost_equal(feats, expected_value, decimal=6)


def test_frame_shift(padding_cut):
    assert padding_cut.frame_shift == 0.01


def test_load_audio(padding_cut):
    samples = padding_cut.load_audio()
    assert samples.shape[0] == 1  # single channel
    assert samples.shape[1] == 160000
    np.testing.assert_equal(samples, 0.0)


@pytest.mark.parametrize(
    [
        "offset",
        "duration",
        "expected_duration",
        "expected_num_frames",
        "expected_num_samples",
    ],
    [
        (0.0, None, 10.0, 1000, 160000),
        (0.0, 5.0, 5.0, 500, 80000),
        (5.0, None, 5.0, 500, 80000),
        (5.0, 5.0, 5.0, 500, 80000),
        (5.0, 2.0, 2.0, 200, 32000),
    ],
)
def test_truncate(
    padding_cut,
    offset,
    duration,
    expected_duration,
    expected_num_frames,
    expected_num_samples,
):
    cut = padding_cut.truncate(offset=offset, duration=duration, preserve_id=True)
    # Invariants
    assert cut.frame_shift == padding_cut.frame_shift
    assert cut.num_features == padding_cut.num_features
    assert cut.sampling_rate == padding_cut.sampling_rate
    assert cut.feat_value == padding_cut.feat_value
    assert cut.id == padding_cut.id
    # Variants
    assert cut.duration == expected_duration
    assert cut.num_frames == expected_num_frames
    assert cut.load_features().shape == (expected_num_frames, 40)
    assert cut.load_audio().shape == (1, expected_num_samples)


@pytest.fixture
def libri_cut():
    return MonoCut(
        channel=0,
        duration=16.04,
        features=Features(
            channels=0,
            duration=16.04,
            num_features=40,
            num_frames=1604,
            frame_shift=0.01,
            recording_id="recording-1",
            sampling_rate=16000,
            start=0.0,
            storage_path="test/fixtures/libri/storage",
            storage_key="30c2440c-93cb-4e83-b382-f2a59b3859b4.llc",
            storage_type="lilcom_files",
            type="fbank",
        ),
        recording=Recording(
            id="recording-1",
            sources=[
                AudioSource(
                    type="file",
                    channels=[0],
                    source="test/fixtures/libri/libri-1088-134315-0000.wav",
                )
            ],
            sampling_rate=16000,
            num_samples=256640,
            duration=1604,
        ),
        id="849e13d8-61a2-4d09-a542-dac1aee1b544",
        start=0.0,
        supervisions=[],
    )


def test_mix_in_the_middle(libri_cut, padding_cut):
    mixed = libri_cut.mix(padding_cut)

    # Invariants
    assert mixed.duration == 16.04
    assert mixed.num_features == 40
    assert mixed.num_frames == 1604

    # Check that the actual feature shapes and values did not change
    pre_mixed_feats = libri_cut.load_features()
    mixed_feats = mixed.load_features()
    assert mixed_feats.shape == pre_mixed_feats.shape
    np.testing.assert_allclose(pre_mixed_feats, mixed_feats, rtol=1e-2)


def test_mix_pad_right(libri_cut, padding_cut):
    mixed = libri_cut.mix(padding_cut, offset_other_by=10.0)

    assert mixed.duration == 20.0
    assert mixed.num_frames == 2000

    mixed_feats = mixed.load_features()
    assert mixed_feats.shape == (2000, 40)
    np.testing.assert_allclose(
        mixed_feats[1604:, :], PADDING_LOG_ENERGY, atol=0.7
    )  # Only padding after 16.04s
    np.testing.assert_array_less(
        PADDING_LOG_ENERGY, mixed_feats[1603, :]
    )  # Padding didn't start before 16.04s

    pre_mixed_feats = libri_cut.load_features()
    np.testing.assert_allclose(pre_mixed_feats, mixed_feats[:1604, :], rtol=1e-2)


def test_mix_pad_left(libri_cut, padding_cut):
    mixed = padding_cut.mix(libri_cut, offset_other_by=3.96)

    assert mixed.duration == 20.0
    assert mixed.num_frames == 2000

    mixed_feats = mixed.load_features()
    assert mixed_feats.shape == (2000, 40)
    np.testing.assert_allclose(
        mixed_feats[:396, :], PADDING_LOG_ENERGY, atol=0.7
    )  # Only padding before 3.96s
    np.testing.assert_array_less(
        PADDING_LOG_ENERGY, mixed_feats[396, :]
    )  # No padding after 3.96s

    pre_mixed_feats = libri_cut.load_features()
    np.testing.assert_allclose(pre_mixed_feats, mixed_feats[396:, :], rtol=1e-2)


@pytest.fixture
def mixed_libri_cut(libri_cut):
    return libri_cut.mix(libri_cut)


def test_mix_mixed_cut_with_padding_in_the_middle(mixed_libri_cut, padding_cut):
    mixed = mixed_libri_cut.mix(padding_cut)

    # Invariants
    assert mixed.duration == 16.04
    assert mixed.num_features == 40
    assert mixed.num_frames == 1604

    # Check that the actual feature shapes and values did not change
    pre_mixed_feats = mixed_libri_cut.load_features()
    mixed_feats = mixed.load_features()
    assert mixed_feats.shape == pre_mixed_feats.shape
    np.testing.assert_allclose(pre_mixed_feats, mixed_feats, rtol=1e-2)


def test_mix_mixed_cut_with_padding_on_the_right(mixed_libri_cut, padding_cut):
    mixed = mixed_libri_cut.mix(padding_cut, offset_other_by=10.0)

    assert mixed.duration == 20.0
    assert mixed.num_frames == 2000

    mixed_feats = mixed.load_features()
    assert mixed_feats.shape == (2000, 40)

    np.testing.assert_allclose(
        mixed_feats[1604:, :], PADDING_LOG_ENERGY, atol=0.8
    )  # Only padding after 16.04s
    np.testing.assert_array_less(
        PADDING_LOG_ENERGY, mixed_feats[1603, :]
    )  # Padding didn't start before 16.04s

    pre_mixed_feats = mixed_libri_cut.load_features()
    np.testing.assert_allclose(pre_mixed_feats, mixed_feats[:1604, :], rtol=1e-1)


def test_mix_mixed_cut_with_padding_on_the_left(mixed_libri_cut, padding_cut):
    mixed = padding_cut.mix(mixed_libri_cut, offset_other_by=3.96)

    assert mixed.duration == 20.0
    assert mixed.num_frames == 2000

    mixed_feats = mixed.load_features()
    assert mixed_feats.shape == (2000, 40)
    np.testing.assert_allclose(
        mixed_feats[:396, :], PADDING_LOG_ENERGY, atol=0.8
    )  # Only padding before 3.96s
    np.testing.assert_array_less(
        PADDING_LOG_ENERGY, mixed_feats[396, :]
    )  # No padding after 3.96s

    pre_mixed_feats = mixed_libri_cut.load_features()
    np.testing.assert_allclose(pre_mixed_feats, mixed_feats[396:, :], rtol=1e-1)


def test_append(libri_cut, padding_cut):
    appended = libri_cut.append(padding_cut)

    assert appended.duration == 26.04
    assert appended.num_frames == 2604

    appended_feats = appended.load_features()
    assert appended_feats.shape == (2604, 40)
    np.testing.assert_allclose(
        appended_feats[1604:, :], PADDING_LOG_ENERGY, atol=0.8
    )  # Only padding after 16.04s
    np.testing.assert_array_less(
        PADDING_LOG_ENERGY, appended_feats[1603, :]
    )  # Padding didn't start before 16.04s

    original_feats = libri_cut.load_features()
    np.testing.assert_allclose(original_feats, appended_feats[:1604, :], rtol=1e-2)


def test_pad_equal_length_does_not_change_cut(libri_cut):
    padded = libri_cut.pad(duration=libri_cut.duration)
    assert padded == libri_cut

    padded = libri_cut.pad(num_frames=libri_cut.num_frames)
    assert padded == libri_cut

    padded = libri_cut.pad(num_samples=libri_cut.num_samples)
    assert padded == libri_cut


@pytest.mark.parametrize(
    [
        "duration",
        "num_frames",
        "num_samples",
        "expected_duration",
        "expected_num_frames",
        "expected_num_samples",
    ],
    [
        (20, None, None, 20, 2000, 320000),
        (None, 2000, None, 20, 2000, 320000),
        (None, None, 320000, 20, 2000, 320000),
    ],
)
def test_pad_simple_cut(
    libri_cut,
    duration,
    num_frames,
    num_samples,
    expected_duration,
    expected_num_frames,
    expected_num_samples,
):
    padded = libri_cut.pad(
        duration=duration, num_frames=num_frames, num_samples=num_samples
    )

    assert padded.duration == expected_duration
    assert padded.num_frames == expected_num_frames
    assert padded.num_samples == expected_num_samples

    mixed_feats = padded.load_features()
    assert mixed_feats.shape == (2000, 40)
    np.testing.assert_allclose(
        mixed_feats[1604:, :], PADDING_LOG_ENERGY, atol=0.8
    )  # Only padding after 16.04s
    np.testing.assert_array_less(
        PADDING_LOG_ENERGY, mixed_feats[1603, :]
    )  # Padding didn't start before 16.04s

    pre_mixed_feats = libri_cut.load_features()
    np.testing.assert_almost_equal(pre_mixed_feats, mixed_feats[:1604, :], decimal=5)


@pytest.mark.parametrize(
    [
        "duration",
        "num_frames",
        "num_samples",
        "expected_duration",
        "expected_num_frames",
        "expected_num_samples",
        "exception_expectation",
    ],
    [
        (20, None, None, 20, 2000, 320000, does_not_raise()),
        (None, 2000, None, 20, 2000, 320000, pytest.raises(AssertionError)),
        (None, None, 320000, 20, 2000, 320000, does_not_raise()),
    ],
)
def test_pad_simple_cut_audio_only(
    libri_cut,
    duration,
    num_frames,
    num_samples,
    expected_duration,
    expected_num_frames,
    expected_num_samples,
    exception_expectation,
):
    libri_cut.features = None
    with exception_expectation:
        padded = libri_cut.pad(
            duration=duration, num_frames=num_frames, num_samples=num_samples
        )

        assert padded.duration == expected_duration
        assert padded.num_samples == expected_num_samples

        mixed_audio = padded.load_audio()
        assert mixed_audio.shape == (1, padded.num_samples)

        pre_mixed_audio = libri_cut.load_audio()
        assert pre_mixed_audio.shape == (1, libri_cut.num_samples)


@pytest.mark.parametrize(
    [
        "duration",
        "num_frames",
        "num_samples",
        "expected_duration",
        "expected_num_frames",
        "expected_num_samples",
        "exception_expectation",
    ],
    [
        (20, None, None, 20, 2000, 320000, does_not_raise()),
        (None, 2000, None, 20, 2000, 320000, does_not_raise()),
        (None, None, 320000, 20, 2000, 320000, pytest.raises(AssertionError)),
    ],
)
def test_pad_simple_cut_features_only(
    libri_cut,
    duration,
    num_frames,
    num_samples,
    expected_duration,
    expected_num_frames,
    expected_num_samples,
    exception_expectation,
):
    libri_cut.recording = None
    with exception_expectation:
        padded = libri_cut.pad(
            duration=duration, num_frames=num_frames, num_samples=num_samples
        )
        assert padded.duration == expected_duration
        assert padded.num_frames == expected_num_frames


@pytest.mark.parametrize(
    [
        "duration",
        "num_frames",
        "num_samples",
        "expected_duration",
        "expected_num_frames",
        "expected_num_samples",
    ],
    [
        (20, None, None, 20, 2000, 320000),
        (None, 2000, None, 20, 2000, 320000),
        (None, None, 320000, 20, 2000, 320000),
    ],
)
def test_pad_mixed_cut(
    mixed_libri_cut,
    duration,
    num_frames,
    num_samples,
    expected_duration,
    expected_num_frames,
    expected_num_samples,
):
    padded = mixed_libri_cut.pad(
        duration=duration, num_frames=num_frames, num_samples=num_samples
    )

    assert padded.duration == expected_duration
    assert padded.num_frames == expected_num_frames
    assert padded.num_samples == expected_num_samples

    mixed_feats = padded.load_features()
    assert mixed_feats.shape == (2000, 40)
    np.testing.assert_allclose(
        mixed_feats[1604:, :], PADDING_LOG_ENERGY, atol=0.8
    )  # Only padding after 16.04s
    np.testing.assert_array_less(
        PADDING_LOG_ENERGY, mixed_feats[1603, :]
    )  # Padding didn't start before 16.04s

    pre_mixed_feats = mixed_libri_cut.load_features()
    np.testing.assert_almost_equal(pre_mixed_feats, mixed_feats[:1604, :], decimal=2)


@pytest.mark.parametrize("direction", ["left", "right", "both"])
@pytest.mark.parametrize(
    [
        "duration",
        "num_frames",
        "num_samples",
        "expected_duration",
        "expected_num_frames",
        "expected_num_samples",
    ],
    [
        (None, None, None, 10.0, 1000, 160000),
        (60.1, None, None, 60.1, 6010, 961600),
        (None, 6010, None, 60.1, 6010, 961600),
        (None, None, 961600, 60.1, 6010, 961600),
    ],
)
def test_pad_cut_set(
    cut_set,
    direction,
    duration,
    num_frames,
    num_samples,
    expected_duration,
    expected_num_frames,
    expected_num_samples,
):
    # cut_set fixture is defined in test/cut/conftest.py
    padded_cut_set = cut_set.pad(
        duration=duration,
        num_frames=num_frames,
        num_samples=num_samples,
        direction=direction,
    )
    assert all(cut.duration == expected_duration for cut in padded_cut_set)
    assert all(cut.num_frames == expected_num_frames for cut in padded_cut_set)
    assert all(cut.num_samples == expected_num_samples for cut in padded_cut_set)


def test_serialize_padded_cut_set(cut_set):
    # cut_set fixture is defined in test/cut/conftest.py
    padded_cut_set = cut_set.pad(60.1)
    with NamedTemporaryFile() as f:
        padded_cut_set.to_json(f.name)
        restored = CutSet.from_json(f.name)
    assert padded_cut_set == restored


def test_pad_without_mixing(libri_cut):
    cut = libri_cut.pad(20)
    cut.tracks[
        0
    ].cut.features.type = (
        "undefined"  # make Lhotse think that a custom feat extractor is used
    )
    cut.load_features()  # does not throw


def test_pad_left_regular_cut(libri_cut):
    cut = libri_cut.pad(30, direction="left")

    assert len(cut.tracks) == 2

    c = cut.tracks[0].cut
    assert isinstance(c, PaddingCut)
    assert c.duration == 13.96

    c = cut.tracks[1].cut
    assert c == libri_cut
    assert cut.tracks[1].offset == 13.96


def test_pad_left_padding_cut(padding_cut):
    cut = padding_cut.pad(30, direction="left")
    assert cut.duration == 30.0


def test_pad_left_mixed_cut(mixed_libri_cut, libri_cut):
    cut = mixed_libri_cut.pad(30, direction="left")

    assert len(cut.tracks) == 3

    c = cut.tracks[0].cut
    assert isinstance(c, PaddingCut)
    assert c.duration == 13.96

    c = cut.tracks[1].cut
    assert c == libri_cut
    assert cut.tracks[1].offset == 13.96

    c = cut.tracks[2].cut
    assert c == libri_cut
    assert cut.tracks[2].offset == 13.96


def test_pad_both_regular_cut(libri_cut):
    cut = libri_cut.pad(30, direction="both")

    assert len(cut.tracks) == 3

    t = cut.tracks[0]
    assert isinstance(t.cut, PaddingCut)
    assert t.offset == 0
    assert t.cut.duration == 6.98

    t = cut.tracks[1]
    assert t.cut == libri_cut
    assert t.offset == 6.98

    t = cut.tracks[2]
    assert isinstance(t.cut, PaddingCut)
    assert t.offset == 23.02
    assert t.cut.duration == 6.98


def test_pad_both_padding_cut(padding_cut):
    cut = padding_cut.pad(30, direction="both")
    assert cut.duration == 30.0
    assert len(cut.tracks) == 3
    assert all(isinstance(t.cut, PaddingCut) for t in cut.tracks)


def test_pad_both_mixed_cut(mixed_libri_cut, libri_cut):
    cut = mixed_libri_cut.pad(30, direction="both")

    assert len(cut.tracks) == 4

    c = cut.tracks[0].cut
    assert isinstance(c, PaddingCut)
    assert c.duration == 6.98

    c = cut.tracks[1].cut
    assert c == libri_cut
    assert cut.tracks[1].offset == 6.98

    c = cut.tracks[2].cut
    assert c == libri_cut
    assert cut.tracks[2].offset == 6.98

    c = cut.tracks[3].cut
    assert isinstance(c, PaddingCut)
    assert c.duration == 6.98
