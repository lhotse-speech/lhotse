from math import isclose

import pytest

from lhotse import CutSet, SupervisionSegment, SupervisionSet
from lhotse.cut import MixedCut, PaddingCut
from lhotse.testing.dummies import (
    dummy_cut,
    dummy_features,
    dummy_recording,
    dummy_temporal_array,
)
from lhotse.utils import LOG_EPSILON, uuid4


@pytest.mark.parametrize(
    [
        "cut_start",
        "cut_duration",
        "extend_duration",
        "extend_direction",
        "pad_silence",
        "expected_start",
        "expected_end",
    ],
    [
        (0.0, 0.5, 0.3, "right", False, 0.0, 0.8),
        (0.0, 0.5, 0.3, "both", False, 0.0, 0.8),
        (0.2, 0.5, 0.3, "left", False, 0.0, 0.7),
        (0.2, 0.5, 0.1, "both", False, 0.1, 0.8),
        (0.0, 0.8, 0.3, "both", False, 0.0, 1.0),
        (0.0, 0.5, 0.3, "right", True, 0.0, 0.8),
        (0.0, 0.5, 0.3, "both", True, 0.0, 1.1),
        (0.2, 0.5, 0.3, "left", True, 0.0, 0.8),
        (0.2, 0.5, 0.1, "both", True, 0.1, 0.8),
        (0.0, 0.8, 0.3, "both", True, 0.0, 1.4),
    ],
)
def test_extend_by_cut(
    cut_start,
    cut_duration,
    extend_duration,
    extend_direction,
    pad_silence,
    expected_start,
    expected_end,
):
    cut = dummy_cut(int(uuid4()), start=cut_start, duration=cut_duration)
    extended_cut = cut.extend_by(
        duration=extend_duration, direction=extend_direction, pad_silence=pad_silence
    )
    assert isclose(extended_cut.start, expected_start)
    assert isclose(extended_cut.end, expected_end)


@pytest.mark.parametrize("preserve_id", [True, False])
def test_extend_by_cut_preserve_id(preserve_id):
    cut = dummy_cut(int(uuid4()), start=0.0, duration=0.5)
    extended_cut = cut.extend_by(
        duration=0.3, direction="right", preserve_id=preserve_id
    )
    if preserve_id:
        assert extended_cut.id == cut.id
    else:
        assert extended_cut.id != cut.id


@pytest.mark.parametrize(
    [
        "cut_start",
        "cut_duration",
        "extend_duration",
        "extend_direction",
        "supervision_start",
        "supervision_duration",
        "expected_start",
        "expected_end",
    ],
    [
        (0.2, 0.5, 0.2, "right", 0.1, 0.7, 0.1, 0.8),
        (0.2, 0.5, 0.1, "both", 0.2, 0.8, 0.3, 1.1),
    ],
)
def test_extend_by_cut_with_supervision(
    cut_start,
    cut_duration,
    extend_duration,
    extend_direction,
    supervision_start,
    supervision_duration,
    expected_start,
    expected_end,
):
    recording = dummy_recording(int(uuid4()), duration=1.0)
    supervisions = SupervisionSet.from_segments(
        [
            SupervisionSegment(
                id=int(uuid4()),
                recording_id=recording.id,
                start=supervision_start,
                duration=supervision_duration,
            )
        ]
    )
    cut = dummy_cut(
        int(uuid4()), start=cut_start, duration=cut_duration, supervisions=supervisions
    )
    extended_cut = cut.extend_by(duration=extend_duration, direction=extend_direction)
    assert isclose(extended_cut.supervisions[0].start, expected_start)
    assert isclose(extended_cut.supervisions[0].end, expected_end)


@pytest.mark.parametrize(
    [
        "cut_start",
        "cut_duration",
        "array_start",
        "extend_duration",
        "extend_direction",
        "expected",
    ],
    [
        (0.3, 0.5, 0.0, 0.1, "right", True),
        (0.3, 0.5, 0.3, 0.8, "right", False),
    ],
)
def test_extend_by_cut_with_temporal_array(
    cut_start,
    cut_duration,
    array_start,
    extend_duration,
    extend_direction,
    expected,
):
    cut = dummy_cut(
        int(uuid4()),
        start=cut_start,
        duration=cut_duration,
        features=None,
        recording=dummy_recording(int(uuid4()), duration=1.5),
    )
    cut.temporal_array = dummy_temporal_array(start=array_start)
    extended_cut = cut.extend_by(duration=extend_duration, direction=extend_direction)
    if expected:
        assert extended_cut.temporal_array == cut.temporal_array
    else:
        with pytest.raises(ValueError):
            _ = extended_cut.load_custom("temporal_array")


@pytest.mark.parametrize(
    [
        "cut_start",
        "cut_duration",
        "feature_start",
        "feature_duration",
        "extend_duration",
        "extend_direction",
        "expected",
    ],
    [
        (0.3, 0.8, 0.0, 1.0, 0.1, "both", True),
        (0.3, 0.8, 0.3, 0.8, 0.1, "both", False),
    ],
)
def test_extend_by_cut_with_features(
    cut_start,
    cut_duration,
    feature_start,
    feature_duration,
    extend_duration,
    extend_direction,
    expected,
):
    cut = dummy_cut(
        int(uuid4()),
        start=cut_start,
        duration=cut_duration,
        features=dummy_features(
            int(uuid4()), start=feature_start, duration=feature_duration
        ),
    )
    extended_cut = cut.extend_by(duration=extend_duration, direction=extend_direction)
    if expected:
        assert extended_cut.features == cut.features
    else:
        assert extended_cut.features is None


def test_cut_set_extend_by():
    cut1 = dummy_cut(int(uuid4()), start=0.0, duration=0.5)
    cut2 = dummy_cut(int(uuid4()), start=0.2, duration=0.4)
    cut_set = CutSet.from_cuts([cut1, cut2])
    extended_cut_set = cut_set.extend_by(
        duration=0.3, direction="both", preserve_id=True
    )
    assert isclose(extended_cut_set[cut1.id].start, 0.0)
    assert isclose(extended_cut_set[cut1.id].end, 0.8)
    assert isclose(extended_cut_set[cut2.id].start, 0.0)
    assert isclose(extended_cut_set[cut2.id].end, 0.9)


# `cut1` and `cut2` parameters are standard Pytest fixtures located in test/cut/conftest.py
def test_mixed_cut_extend_by(cut1, cut2):
    mixed_cut = cut1.mix(cut2, offset_other_by=1.0)
    assert isinstance(mixed_cut, MixedCut)
    with pytest.raises(ValueError):
        mixed_cut.extend_by(duration=0.3, direction="both", preserve_id=True)


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
        feat_value=LOG_EPSILON,
    )


@pytest.mark.parametrize(
    [
        "extend_duration",
        "extend_direction",
        "expected_duration",
        "expected_num_frames",
    ],
    [(0.3, "right", 10.3, 1030), (0.5, "both", 11.0, 1100)],
)
def test_padding_cut_extend_by(
    padding_cut,
    extend_duration,
    extend_direction,
    expected_duration,
    expected_num_frames,
):
    padding_cut_extended = padding_cut.extend_by(
        duration=extend_duration, direction=extend_direction
    )
    assert isinstance(padding_cut_extended, PaddingCut)
    # Invariants
    assert padding_cut.frame_shift == padding_cut_extended.frame_shift
    assert padding_cut.num_features == padding_cut_extended.num_features
    assert padding_cut.sampling_rate == padding_cut_extended.sampling_rate
    assert padding_cut.feat_value == padding_cut_extended.feat_value
    # Variants
    assert padding_cut_extended.duration == expected_duration
    assert padding_cut_extended.num_frames == expected_num_frames
