__all__ = (
    "Activity",
    "ActivityDetector",
    "check_detetor",
)
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np


@dataclass
class Activity:
    start: float
    duration: float


class ActivityDetector(ABC):
    detector_name: str = "base"
    sampling_rate: int = 0
    _known_detectors: Dict[str, Type["ActivityDetector"]] = {}

    def __init__(self, device: str = "cpu"):
        self._device = device

    def __init_subclass__(
        cls,
        detector_name: Optional[str] = None,
        sampling_rate: int = 0,
    ) -> None:
        detector_name = detector_name.lower().replace("-", "_")
        cls.detector_name = detector_name or cls.__name__
        cls.sampling_rate = sampling_rate
        cls._known_detectors[cls.detector_name] = cls
        # TODO: check detector __init__ signature

    @property
    def device(self) -> str:
        return self._device

    @abstractmethod
    def __call__(self, track: np.ndarray) -> List[Activity]:  # pragma: no cover
        raise NotImplementedError()

    @classmethod
    def hard_reset(cls) -> None:  # pragma: no cover
        """Do some work for preloading / resetting the model state."""
        return None

    @classmethod
    def list_detectors(cls) -> Tuple[str, ...]:
        """List the names of all known activity detectors."""
        return tuple(name for name in cls._known_detectors if not name.startswith("_"))

    @classmethod
    def get_detector(
        cls,
        detector: Union[str, Type["ActivityDetector"]],
    ) -> Type["ActivityDetector"]:
        """Get a specific activity detector by name."""
        if isinstance(detector, str):
            detector = detector.lower().replace("-", "_")
            detector_kls = cls._known_detectors.get(detector)
            if detector_kls is None:
                raise ValueError(f"No such activity detector: {detector}")
            return detector_kls
        if not issubclass(detector, ActivityDetector):  # type: ignore
            raise ValueError(f"Expected an ActivityDetector subclass, got {detector}")
        return detector


def check_detetor(detector: Union[str, Type[ActivityDetector]], force: bool = False):
    if force:  # pragma: no cover
        print("Removing model state from cache...")
        ActivityDetector.get_detector(detector).hard_reset()
    else:
        print("Checking model state in cache...")
        ActivityDetector.get_detector(detector)(device="cpu")
