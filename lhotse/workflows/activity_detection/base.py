import abc
from dataclasses import dataclass
from typing import List

import numpy as np

from lhotse.audio.recording import Recording
from lhotse.supervision import SupervisionSegment


@dataclass
class Activity:
    start: float
    duration: float


class ActivityDetector(abc.ABC):
    def __init__(self, detector_name: str, sampling_rate: int, device: str = "cpu"):
        self._detector_name = detector_name
        self._sampling_rate = sampling_rate
        self._device = device

    @property
    def device(self) -> str:
        return self._device

    def __call__(self, recording: Recording) -> List[SupervisionSegment]:
        resampled = recording.resample(self._sampling_rate)
        audio = resampled.load_audio()  # type: ignore

        uid_template = "{recording_id}-{detector_name}-{channel}-{number:05}"

        result: List[SupervisionSegment] = []
        for channel, track in enumerate(audio):
            track = np.squeeze(track)
            activities = self.forward(track)

            for i, activity in enumerate(activities):
                uid = uid_template.format(
                    recording_id=recording.id,
                    detector_name=self._detector_name,
                    channel=channel,
                    number=i,
                )
                segment = SupervisionSegment(
                    id=uid,
                    recording_id=recording.id,
                    start=activity.start,
                    duration=activity.duration,
                    channel=channel,
                )
                result.append(segment)

        return result

    @abc.abstractmethod
    def forward(self, track: np.ndarray) -> List[Activity]:  # pragma: no cover
        raise NotImplementedError()

    @classmethod
    def force_download(cls):  # pragma: no cover
        """Do some work for preloading / resetting the model state."""
        pass
