import abc
from typing import List

from lhotse.audio.recording import Recording
from lhotse.supervision import AlignmentItem


class ActivityDetector(abc.ABC):
    def __init__(self, device: str = "cpu"):
        self._device = device

    @property
    def device(self) -> str:
        return self._device

    @abc.abstractmethod
    def __call__(self, recording: Recording) -> List[AlignmentItem]:
        pass
