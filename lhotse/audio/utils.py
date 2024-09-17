import functools
import logging
import os
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Optional

from lhotse.utils import NonPositiveEnergyError, Seconds, fastcopy, suppress_and_warn

_DEFAULT_LHOTSE_AUDIO_DURATION_MISMATCH_TOLERANCE: Seconds = 0.025
_LHOTSE_AUDIO_DURATION_MISMATCH_TOLERANCE: Seconds = (
    _DEFAULT_LHOTSE_AUDIO_DURATION_MISMATCH_TOLERANCE
)


@dataclass
class VideoInfo:
    """
    Metadata about video content in a :class:`~lhotse.audio.Recording`.
    """

    fps: float
    """Video frame rate (frames per second). It's a float because some standard FPS are fractional (e.g. 59.94)"""

    num_frames: int
    """Number of video frames."""

    height: int
    """Height in pixels."""

    width: int
    """Width in pixels."""

    @property
    def duration(self) -> Seconds:
        return self.num_frames / self.fps

    @property
    def frame_length(self) -> Seconds:
        return 1.0 / self.fps

    def copy_with(self, **kwargs) -> "VideoInfo":
        return fastcopy(self, **kwargs)


def get_audio_duration_mismatch_tolerance() -> Seconds:
    """Retrieve the current audio duration mismatch tolerance in seconds."""
    if (
        _LHOTSE_AUDIO_DURATION_MISMATCH_TOLERANCE
        != _DEFAULT_LHOTSE_AUDIO_DURATION_MISMATCH_TOLERANCE
    ):
        return _LHOTSE_AUDIO_DURATION_MISMATCH_TOLERANCE

    if "LHOTSE_AUDIO_DURATION_MISMATCH_TOLERANCE" in os.environ:
        return float(os.environ["LHOTSE_AUDIO_DURATION_MISMATCH_TOLERANCE"])

    return _LHOTSE_AUDIO_DURATION_MISMATCH_TOLERANCE


def set_audio_duration_mismatch_tolerance(delta: Seconds) -> None:
    """
    Override Lhotse's global threshold for allowed audio duration mismatch between the
    manifest and the actual data.

    Some scenarios when a mismatch can happen:

        - the :class:`.Recording` manifest duration is rounded off too much
            (i.e., bad user input, but too inconvenient to go back and fix the manifests)

        - data augmentation changes the number of samples a bit in a difficult to predict way

    When there is a mismatch, Lhotse will either trim or replicate the diff to match
    the value found in the :class:`.Recording` manifest.

    .. note:: We don't recommend setting this lower than the default value, as it could
        break some data augmentation transforms.

    Example::

        >>> import lhotse
        >>> lhotse.set_audio_duration_mismatch_tolerance(0.01)  # 10ms tolerance

    :param delta: New tolerance in seconds.
    """
    global _LHOTSE_AUDIO_DURATION_MISMATCH_TOLERANCE
    logging.info(
        "The user overrided tolerance for audio duration mismatch "
        "between the values in the manifest and the actual data. "
        f"Old threshold: {_LHOTSE_AUDIO_DURATION_MISMATCH_TOLERANCE}s. "
        f"New threshold: {delta}s."
    )
    if delta < _DEFAULT_LHOTSE_AUDIO_DURATION_MISMATCH_TOLERANCE:
        warnings.warn(
            "The audio duration mismatch tolerance has been set to a value lower than "
            f"default ({_DEFAULT_LHOTSE_AUDIO_DURATION_MISMATCH_TOLERANCE}s). "
            f"We don't recommend this as it might break some data augmentation transforms."
        )
    _LHOTSE_AUDIO_DURATION_MISMATCH_TOLERANCE = delta


class VideoLoadingError(Exception):
    pass


class AudioLoadingError(Exception):
    pass


class AudioSavingError(Exception):
    pass


class DurationMismatchError(Exception):
    pass


@contextmanager
def suppress_audio_loading_errors(enabled: bool = True):
    """
    Context manager that suppresses errors related to audio loading.
    Emits warning to the console.
    """
    with suppress_and_warn(
        AudioLoadingError,
        DurationMismatchError,
        NonPositiveEnergyError,
        ConnectionResetError,  # when reading from object stores / network sources
        enabled=enabled,
    ):
        yield


@contextmanager
def suppress_video_loading_errors(enabled: bool = True):
    """
    Context manager that suppresses errors related to audio loading.
    Emits warning to the console.
    """
    with suppress_and_warn(
        VideoLoadingError,
        AudioLoadingError,
        DurationMismatchError,
        NonPositiveEnergyError,
        ConnectionResetError,  # when reading from object stores / network sources
        enabled=enabled,
    ):
        yield


def null_result_on_audio_loading_error(func: Callable) -> Callable:
    """
    This is a decorator that makes a function return None when reading audio with Lhotse failed.

    Example::

        >>> @null_result_on_audio_loading_error
        ... def func_loading_audio(rec):
        ...     audio = rec.load_audio()  # if this fails, will return None instead
        ...     return other_func(audio)

    Another example::

        >>> # crashes on loading audio
        >>> audio = load_audio(cut)
        >>> # does not crash on loading audio, return None instead
        >>> maybe_audio: Optional = null_result_on_audio_loading_error(load_audio)(cut)
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Optional:
        with suppress_audio_loading_errors():
            return func(*args, **kwargs)

    return wrapper


def verbose_audio_loading_exceptions() -> bool:
    return os.environ.get("LHOTSE_AUDIO_LOADING_EXCEPTION_VERBOSE") == "1"
