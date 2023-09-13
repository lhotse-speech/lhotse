import random

import numpy as np
import pytest
import torch


@pytest.fixture
def deterministic_rng():
    """
    Pytest fixture that ensures deterministic RNG behavior.
    After the test finishes, it restores the previous RNG state.

    Example usage::

        >>> def my_test(deterministic_rng):
        ...     x = torch.randn(10, 5)  # always has the same values

    .. note: Learn more about pytest fixtures setup/teardown here:
        https://docs.pytest.org/en/latest/how-to/fixtures.html#teardown-cleanup-aka-fixture-finalization
    """
    SEED = 0

    torch_state = torch.get_rng_state()
    np_state = np.random.get_state()
    py_state = random.getstate()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    yield SEED

    random.setstate(py_state)
    np.random.set_state(np_state)
    torch.set_rng_state(torch_state)
