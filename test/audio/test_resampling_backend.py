import contextlib
import ctypes
import os
import sys

import numpy as np
import pytest

import lhotse
import lhotse.tools.libsox
from lhotse.audio.resampling_backend import (
    available_resampling_backends,
    get_resampling_backend,
    resampling_backend,
    set_resampling_backend,
)


@pytest.fixture(autouse=True)
def reload_module():
    if "lhotse.tools.libsox" in sys.modules:
        del sys.modules["lhotse.tools.libsox"]

    import lhotse.tools.libsox


def test_available_resampling_backends():
    assert isinstance(available_resampling_backends(), list)
    assert len(available_resampling_backends()) > 0
    assert "default" in available_resampling_backends()
    assert "sox" in available_resampling_backends()


def test_get_resampling_backend():
    assert lhotse.get_resampling_backend() in available_resampling_backends()


def test_set_resampling_backend():
    for backend in available_resampling_backends():
        set_resampling_backend(backend)
        assert lhotse.get_resampling_backend() == backend


def test_resampling_backend_contextmanager():
    for backend in available_resampling_backends():
        with resampling_backend(backend):
            assert lhotse.get_resampling_backend() == backend


@contextlib.contextmanager
def monkeypatch_ctypes_util_find_library_so_sox_is_not_available():
    original_find_library = ctypes.util.find_library

    def find_library(name):
        if name == "sox":
            return None
        return original_find_library(name)

    ctypes.util.find_library = find_library
    yield
    ctypes.util.find_library = original_find_library


def test_resampling_backend_contextmanager_sox_works_even_if_sox_is_not_available():
    with monkeypatch_ctypes_util_find_library_so_sox_is_not_available():
        with resampling_backend("sox"):
            assert (
                lhotse.get_resampling_backend() == "sox"
            ), "Sox should be set as the resample backend"


def test_libsox_backend():
    if not lhotse.tools.libsox.libsox_available():
        pytest.skip("Sox is not available")

    with resampling_backend("sox"):
        assert lhotse.get_resampling_backend() == "sox"

        assert (
            lhotse.tools.libsox.LIBSOX_INITIALIZED == False
        ), "Sox should not be initialized before first resampling"

        signal, sampling_rate = np.zeros(16000, dtype=np.float32), 16000
        target_sampling_rate = 8000
        resampled_signal, new_sampling_rate = lhotse.tools.libsox.libsox_rate(
            signal, sampling_rate, target_sampling_rate
        )
        assert resampled_signal.shape == (8000,)
        assert new_sampling_rate == 8000

        assert (
            lhotse.tools.libsox.LIBSOX_INITIALIZED == True
        ), "Sox should be initialized after a resampling"

    assert (
        lhotse.tools.libsox.LIBSOX_INITIALIZED == True
    ), "Sox should be initialized after the context manager is exited"


def test_libsox_backend_not_available():
    with resampling_backend("sox"):
        with monkeypatch_ctypes_util_find_library_so_sox_is_not_available():
            assert (
                lhotse.tools.libsox.LIBSOX_INITIALIZED == False
            ), "Sox should not be initialized before first resampling"
            with pytest.raises(RuntimeError):
                lhotse.tools.libsox.libsox_rate(
                    np.zeros(16000, dtype=np.float32), 16000, 8000
                )
            assert (
                lhotse.tools.libsox.LIBSOX_INITIALIZED == False
            ), "Sox should not be initialized because implicit initialization is expected to fail"
            with pytest.raises(RuntimeError):
                lhotse.tools.libsox.libsox_import()
            assert (
                lhotse.tools.libsox.LIBSOX_INITIALIZED == False
            ), "Sox should not be initialized because explicit initialization is expected to fail"
