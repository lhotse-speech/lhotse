from math import isclose

import pytest

from lhotse.utils import uuid4
from lhotse.testing.dummies import dummy_features, dummy_cut


@pytest.mark.parametrize(
    [
        "cut_start",
        "cut_duration",
        "extend_duration",
        "extend_direction",
        "expected_start",
        "expected_end",
    ],
    [
        (0.0, 0.5, 0.3, "right", 0.0, 0.8),
        (0.0, 0.5, 0.3, "both", 0.0, 0.8),
        (0.2, 0.5, 0.3, "left", 0.0, 0.7),
        (0.2, 0.5, 0.1, "both", 0.1, 0.8),
        (0.0, 0.8, 0.3, "both", 0.0, 1.0),
    ],
)
def test_extend_cut(
    cut_start,
    cut_duration,
    extend_duration,
    extend_direction,
    expected_start,
    expected_end,
):
    cut = dummy_cut(int(uuid4()), start=cut_start, duration=cut_duration)
    extended_cut = cut.extend(duration=extend_duration, direction=extend_direction)
    assert isclose(extended_cut.start, expected_start)
    assert isclose(extended_cut.end, expected_end)


@pytest.mark.parametrize("preserve_id", [True, False])
def test_extend_cut_preserve_id(preserve_id):
    cut = dummy_cut(int(uuid4()), start=0.0, duration=0.5)
    extended_cut = cut.extend(duration=0.3, direction="right", preserve_id=preserve_id)
    if preserve_id:
        assert extended_cut.id == cut.id
    else:
        assert extended_cut.id != cut.id


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
def test_extend_cut_with_features(
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
    extended_cut = cut.extend(duration=extend_duration, direction=extend_direction)
    if expected:
        assert extended_cut.features == cut.features
    else:
        assert extended_cut.features is None
