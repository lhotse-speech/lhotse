import pytest

from lhotse.parallel import parallel_map


def pow2(x):
    return x**2


def mul(x, y):
    return x * y


@pytest.mark.parametrize("num_jobs", [1, 2])
def test_parallel_map_num_jobs(num_jobs):
    squares = list(map(pow2, range(100)))
    squares_parallel = list(parallel_map(pow2, range(100), num_jobs=num_jobs))
    assert squares == squares_parallel


def test_parallel_map_threads():
    squares = list(map(pow2, range(100)))
    squares_parallel = list(parallel_map(pow2, range(100), num_jobs=2, threads=True))
    assert squares == squares_parallel


def test_parallel_map_two_iterables():
    squares = list(map(mul, range(100), range(100)))
    squares_parallel = list(parallel_map(mul, range(100), range(100), num_jobs=2))
    assert squares == squares_parallel
