import pytest

from lhotse.utils import TimeSpan, measure_overlap


@pytest.mark.parametrize(
    ["lhs", "rhs", "expected"],
    [
        (TimeSpan(0, 10), TimeSpan(2, 3), 1.0),
        (TimeSpan(2, 3), TimeSpan(0, 10), 1.0),
        (TimeSpan(0, 10), TimeSpan(0, 10), 1.0),
        (TimeSpan(0, 10), TimeSpan(5, 15), 0.5),
        (TimeSpan(0, 4), TimeSpan(1, 10), 0.75),
        (TimeSpan(0, 1), TimeSpan(1, 2), 0.0),
        (TimeSpan(0, 1), TimeSpan(2, 3), 0.0),
    ],
)
def test_measure_overlap(lhs, rhs, expected):
    assert measure_overlap(lhs, rhs) == pytest.approx(expected)
