from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Activity:
    start: float
    duration: float


class ActivityDetector(ABC):
    def __init__(self, detector_name: str, sampling_rate: int, device: str = "cpu"):
        self._detector_name = detector_name
        self._sampling_rate = sampling_rate
        self._device = device

    @property
    def name(self) -> str:
        return self._detector_name

    @property
    def device(self) -> str:
        return self._device

    @property
    def sampling_rate(self) -> int:
        return self._sampling_rate

    @abstractmethod
    def __call__(self, track: np.ndarray) -> List[Activity]:  # pragma: no cover
        raise NotImplementedError()

    @classmethod
    def force_download(cls) -> None:  # pragma: no cover
        """Do some work for preloading / resetting the model state."""
        return None
