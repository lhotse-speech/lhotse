import random

import numpy as np
import pytest
import torch


@pytest.fixture
def deterministic_rng(request):
    """
    Pytest fixture that ensures deterministic RNG behavior.
    After the test finishes, it restores the previous RNG state.

    Example usage::

        >>> def my_test(deterministic_rng):
        ...     x = torch.randn(10, 5)  # always has the same values

    You can also set random seed like this::

        >>> @pytest.mark.seed(1337)
        ... def my_test(deterministic_rng):
        ...     x = torch.randn(10, 5)

    .. note: Learn more about pytest fixtures setup/teardown here:
        https://docs.pytest.org/en/latest/how-to/fixtures.html#teardown-cleanup-aka-fixture-finalization
    """

    # The mechanism below is pytest's way of parameterizing fixtures.
    # We use that to optionally sed a different random seed than the default 0.
    # See: https://docs.pytest.org/en/7.1.x/how-to/fixtures.html#using-markers-to-pass-data-to-fixtures
    marker = request.node.get_closest_marker("seed")
    if marker is None:
        # Handle missing marker in some way...
        SEED = 0
    else:
        SEED = marker.args[0]

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
