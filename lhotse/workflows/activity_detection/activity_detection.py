from itertools import chain
from typing import List

from lhotse.audio import Recording
from lhotse.supervision import SupervisionSegment, SupervisionSet

from .base import Processor, ProcessWorker


def _detect_acitvity(recording: Recording, model) -> List[SupervisionSegment]:
    return model(recording)


def detect_activity(
    recordings: Recording,
    detector_kls,
    num_jobs: int,
    verbose: bool = False,
) -> SupervisionSet:
    # TODO: work in parallel here
    worker = ProcessWorker(
        gen_model=detector_kls,
        do_work=_detect_acitvity,
        warnings_mode="ignore",
    )
    processor = Processor(
        worker, num_jobs=num_jobs, verbose=verbose, mp_context="spawn"
    )
    segments = chain.from_iterable(processor(recordings))
    supervisions = SupervisionSet.from_segments(segments)

    return supervisions
