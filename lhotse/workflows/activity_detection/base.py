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


def _format_sampling_rate(sampling_rate: int) -> str:
    formatted_rate = str(round(sampling_rate / 1000, 1))
    formatted_rate = formatted_rate.replace(".0", "")
    return f"{formatted_rate}kHz"


class ActivityDetector(abc.ABC):
    def __init__(self, detector_name: str, sampling_rate: int, device: str = "cpu"):
        self._name = f"{detector_name}_{_format_sampling_rate(sampling_rate)}"
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
                    detector_name=self._name,
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
    def forward(self, track: np.ndarray) -> List[Activity]:
        raise NotImplementedError()


class ActivityDetectionProcessor:
    _detectors: Dict[Optional[int], ActivityDetector] = {}

    def __init__(
        self,
        detector_kls: Type[ActivityDetector],
        num_jobs: int,
        device: str = "cpu",
        verbose: bool = False,
    ):
        self._make_detecor = partial(detector_kls, device=device)
        self._num_jobs = num_jobs
        self._verbose = verbose

    def _init_detector(self):
        pid = multiprocessing.current_process().pid
        self._detectors[pid] = self._make_detecor()

    def _process_recording(self, record: Recording) -> List[SupervisionSegment]:
        pid = multiprocessing.current_process().pid
        detector = self._detectors[pid]
        return detector(record)

    def __call__(self, recordings: RecordingSet) -> SupervisionSet:
        start_method = multiprocessing.get_start_method()

        pool = ProcessPoolExecutor(
            max_workers=self._num_jobs,
            initializer=self._init_detector,
            mp_context=multiprocessing.get_context(start_method),
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
            except KeyboardInterrupt as exc:
                pool.shutdown(wait=False)
                if self._verbose:
                    print("Activity detection interrupted by the user.")
                raise exc
            except Exception as exc:
                pool.shutdown(wait=False)
                raise RuntimeError(
                    "Activity detection failed. Please report this issue."
                ) from exc
            finally:
                self._detectors.clear()
