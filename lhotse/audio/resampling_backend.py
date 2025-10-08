import contextlib
import os
import typing
from typing import List, Literal, Optional, Union

ResamplingBackend = Literal["default", "sox"]
CURRENT_RESAMPLING_BACKEND: Optional[ResamplingBackend] = None


def set_current_resampling_backend(backend: ResamplingBackend) -> None:
    global CURRENT_RESAMPLING_BACKEND

    if backend not in available_resampling_backends():
        raise ValueError(
            f"Invalid resample backend: {backend}. Available backends: {available_resampling_backends()}"
        )

    CURRENT_RESAMPLING_BACKEND = backend


def get_current_resampling_backend() -> ResamplingBackend:
    global CURRENT_RESAMPLING_BACKEND

    if CURRENT_RESAMPLING_BACKEND is not None:
        return CURRENT_RESAMPLING_BACKEND

    maybe_env_backend = os.environ.get("LHOTSE_RESAMPLING_BACKEND")
    if maybe_env_backend:
        set_current_resampling_backend(maybe_env_backend)
        return CURRENT_RESAMPLING_BACKEND

    set_current_resampling_backend("default")
    return CURRENT_RESAMPLING_BACKEND


def available_resampling_backends() -> List[ResamplingBackend]:
    return list(typing.get_args(ResamplingBackend))


@contextlib.contextmanager
def resampling_backend(backend: Union[ResamplingBackend, str]):
    previous_backend = get_current_resampling_backend()
    set_current_resampling_backend(backend)
    yield
    set_current_resampling_backend(previous_backend)
