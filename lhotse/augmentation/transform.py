from dataclasses import asdict
from typing import Optional, Tuple

import numpy as np

from lhotse.utils import Seconds


class AudioTransform:
    """
    Base class for all audio transforms that are going to be lazily applied on
    ``Recording`` during loading the audio into memory.

    Any ``AudioTransform`` can be used like a Python function, that expects two arguments:
    a numpy array of samples, and a sampling rate. E.g.:

        >>> fn = AudioTransform.from_dict(...)
        >>> new_audio = fn(audio, sampling_rate)

    Since we often use cuts of the original recording, they will refer to the timestamps
    of the augmented audio (which might be speed perturbed and of different duration).
    Each transform provides a helper method to recover the original audio timestamps:

        >>> # When fn does speed perturbation:
        >>> fn.reverse_timestamps(offset=5.055555, duration=10.1111111, sampling_rate=16000)
        ... (5.0, 10.0)

    Furthermore, ``AudioTransform`` can be easily (de)serialized to/from dict
    that contains its name and parameters.
    This enables storing recording and cut manifests with the transform info
    inside, avoiding the need to store the augmented recording version on disk.

    All audio transforms derived from this class are "automagically" registered,
    so that ``AudioTransform.from_dict()`` can "find" the right type given its name
    to instantiate a specific transform object.
    All child classes are expected to be decorated with a ``@dataclass`` decorator.
    """

    KNOWN_TRANSFORMS = {}

    def __init_subclass__(cls, **kwargs):
        if cls.__name__ not in AudioTransform.KNOWN_TRANSFORMS:
            AudioTransform.KNOWN_TRANSFORMS[cls.__name__] = cls
        super().__init_subclass__(**kwargs)

    def to_dict(self) -> dict:
        data = asdict(self)
        return {"name": type(self).__name__, "kwargs": data}

    @staticmethod
    def from_dict(data: dict) -> "AudioTransform":
        assert (
            data["name"] in AudioTransform.KNOWN_TRANSFORMS
        ), f"Unknown transform type: {data['name']}"
        return AudioTransform.KNOWN_TRANSFORMS[data["name"]](**data["kwargs"])

    def __call__(self, samples: np.ndarray, sampling_rate: int) -> np.ndarray:
        """
        Apply transform.

        To be implemented in derived classes.
        """
        raise NotImplementedError

    def reverse_timestamps(
        self, offset: Seconds, duration: Optional[Seconds], sampling_rate: int
    ) -> Tuple[Seconds, Optional[Seconds]]:
        """
        Convert ``offset`` and ``duration`` timestamps to be adequate for the audio before the transform.
        Useful for on-the-fly augmentation when a particular chunk of audio needs to be read from disk.

        To be implemented in derived classes.
        """
        raise NotImplementedError
