import abc
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from itertools import chain
from typing import Dict, List, Optional, Type

import numpy as np

from lhotse.audio.recording import Recording
from lhotse.audio.recording_set import RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet


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


class ActivityDetectionProcessor:
    _detectors: Dict[Optional[int], ActivityDetector] = {}

    def __init__(
        self,
        detector_kls: Type[ActivityDetector],
        num_jobs: int,
        device: str = "cpu",
        verbose: bool = False,
    ):
        self._make_detector = partial(detector_kls, device=device)
        self._num_jobs = num_jobs
        self._verbose = verbose

    def _init_detector(self):
        pid = multiprocessing.current_process().pid
        self._detectors[pid] = self._make_detector()

    def _process_recording(self, record: Recording) -> List[SupervisionSegment]:
        pid = multiprocessing.current_process().pid
        detector = self._detectors[pid]
        return detector(record)

    def __call__(self, recordings: RecordingSet) -> SupervisionSet:
        pool = ProcessPoolExecutor(
            max_workers=self._num_jobs,
            initializer=self._init_detector,
            mp_context=multiprocessing.get_context("spawn"),
        )

        with pool as executor:
            try:
                parts = executor.map(self._process_recording, recordings)
                if self._verbose:
                    from tqdm.auto import tqdm

                    parts = tqdm(
                        parts,
                        total=len(recordings),
                        desc="Detecting activities",
                        unit="rec",
                    )
                segments = chain.from_iterable(parts)
                return SupervisionSet.from_segments(segments)
            except KeyboardInterrupt as exc:  # pragma: no cover
                pool.shutdown(wait=False)
                if self._verbose:
                    print("Activity detection interrupted by the user.")
                raise exc
            except Exception as exc:  # pragma: no cover
                pool.shutdown(wait=False)
                raise RuntimeError(
                    "Activity detection failed. Please report this issue."
                ) from exc
            finally:
                self._detectors.clear()
