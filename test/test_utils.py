from pytest import mark

from lhotse.utils import overlaps, TimeSpan, overspans


@mark.parametrize(
    ['lhs', 'rhs', 'expected'], [
        (TimeSpan(0, 1), TimeSpan(0, 1), True),
        (TimeSpan(0.5, 1), TimeSpan(0, 1), True),
        (TimeSpan(0, 0.5), TimeSpan(0, 1), True),
        (TimeSpan(0.1, 0.9), TimeSpan(0, 1), True),
        (TimeSpan(1, 2), TimeSpan(0, 1), False),
        (TimeSpan(2, 3), TimeSpan(0, 1), False),
        (TimeSpan(1, 1), TimeSpan(0, 1), False),
        (TimeSpan(0, 0), TimeSpan(0, 1), False),
    ]
)
def test_overlaps(lhs, rhs, expected):
    assert overlaps(lhs, rhs) == expected
    assert overlaps(rhs, lhs) == expected


@mark.parametrize(
    ['lhs', 'rhs', 'expected'], [
        (TimeSpan(0, 1), TimeSpan(0, 1), True),
        (TimeSpan(0.5, 1), TimeSpan(0, 1), False),
        (TimeSpan(0, 1), TimeSpan(0.5, 1), True),
        (TimeSpan(0, 0.5), TimeSpan(0, 1), False),
        (TimeSpan(0, 1), TimeSpan(0, 0.5), True),
        (TimeSpan(0.1, 0.9), TimeSpan(0, 1), False),
        (TimeSpan(0, 1), TimeSpan(0.1, 0.9), True),
        (TimeSpan(1, 2), TimeSpan(0, 1), False),
        (TimeSpan(0, 1), TimeSpan(0, 2), False),
        (TimeSpan(2, 3), TimeSpan(0, 1), False),
        (TimeSpan(0, 1), TimeSpan(2, 3), False),
        (TimeSpan(1, 1), TimeSpan(0, 1), False),
        (TimeSpan(0, 1), TimeSpan(1, 1), True),
        (TimeSpan(0, 0), TimeSpan(0, 1), False),
        (TimeSpan(0, 1), TimeSpan(0, 0), True),
    ]
)
def test_overspans(lhs, rhs, expected):
    assert overspans(lhs, rhs) == expected
