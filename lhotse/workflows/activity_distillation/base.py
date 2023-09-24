import abc
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import chain
from typing import Dict, List, Optional, Type

from lhotse.audio.recording import Recording
from lhotse.audio.recording_set import RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet


class ActivityDetector(abc.ABC):
    def __init__(self, device: str = "cpu"):
        self._device = device

    @property
    def device(self) -> str:
        return self._device

    @abc.abstractmethod
    def __call__(self, recording: Recording) -> List[SupervisionSegment]:
        pass


DETECTORS: Dict[Optional[int], ActivityDetector] = {}


class ActivityDetectionProcessor:
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
        DETECTORS[pid] = self._make_detecor()

    def _process_recording(self, record: Recording) -> List[SupervisionSegment]:
        pid = multiprocessing.current_process().pid
        detector = DETECTORS[pid]
        result = detector(record)
        return result

    def __call__(self, recordings: RecordingSet) -> SupervisionSet:
        pool = ProcessPoolExecutor(
            max_workers=self._num_jobs,
            initializer=self._init_detector,
            # mp_context=multiprocessing.get_context("spawn"),
        )

        with pool as executor:
            parts = executor.map(self._process_recording, recordings)

        segments = chain.from_iterable(parts)
        return SupervisionSet.from_segments(segments)
