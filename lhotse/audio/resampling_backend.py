import contextlib
import os
import typing
from typing import List, Literal

ResamplingBackend = Literal["default", "sox"]
CURRENT_RESAMPLING_BACKEND: ResamplingBackend = "default"


def set_resampling_backend(backend: ResamplingBackend) -> None:
    global CURRENT_RESAMPLING_BACKEND

    if backend not in available_resampling_backends():
        raise ValueError(
            f"Invalid resample backend: {backend}. Available backends: {available_resampling_backends()}"
        )

    CURRENT_RESAMPLING_BACKEND = backend


def get_resampling_backend() -> ResamplingBackend:
    return CURRENT_RESAMPLING_BACKEND


def set_resampling_backend_from_env():
    if os.environ.get("LHOTSE_RESAMPLING_BACKEND"):
        set_resampling_backend(os.environ.get("LHOTSE_RESAMPLING_BACKEND"))


def available_resampling_backends() -> List[ResamplingBackend]:
    return list(typing.get_args(ResamplingBackend))


@contextlib.contextmanager
def resampling_backend(backend: ResamplingBackend | str):
    previous_backend = get_resampling_backend()
    set_resampling_backend(backend)
    yield
    set_resampling_backend(previous_backend)
