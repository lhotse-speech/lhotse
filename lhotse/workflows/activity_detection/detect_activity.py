__all__ = (
    "detect_acitvity_segments",
    "detect_activity",
)

from functools import partial
from itertools import chain
from typing import Iterable, List, Optional, Type, Union

import numpy as np

from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.workflows.backend import Processor, ProcessWorker

from ._tools import PathLike, as_path, assert_output_file
from .base import ActivityDetector


def detect_acitvity_segments(
    recording: Recording,
    model: ActivityDetector,
) -> List[SupervisionSegment]:
    resampled = recording.resample(model.sampling_rate)
    audio = resampled.load_audio()  # type: ignore

    uid_template = "{recording_id}-{detector_name}-{channel}-{number:05}"

    result: List[SupervisionSegment] = []
    for channel, track in enumerate(audio):
        track = np.squeeze(track)
        activities = model(track)

        for i, activity in enumerate(activities):
            uid = uid_template.format(
                recording_id=recording.id,
                detector_name=model.name,
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


def detect_activity(
    recordings: Union[Iterable[Recording], PathLike],
    detector: Union[str, Type[ActivityDetector]],
    # output
    output_supervisions_manifest: Optional[PathLike] = None,
    # mode
    num_jobs: int = 1,
    verbose: bool = False,
    device: str = "cpu",
) -> SupervisionSet:
    if path := as_path(recordings):
        if verbose:
            print(f"Loading recordings from {path}...")
        recordings = RecordingSet.from_file(str(path))

    output_supervisions_manifest = assert_output_file(
        output_supervisions_manifest, "output_supervisions_manifest"
    )

    detector_kls = ActivityDetector.get_detector(detector)
    worker = ProcessWorker(
        gen_model=partial(detector_kls, device=device),
        do_work=detect_acitvity_segments,
        warnings_mode="ignore",
    )
    processor = Processor(
        worker, num_jobs=num_jobs, verbose=verbose, mp_context="spawn"
    )
    segments = chain.from_iterable(processor(recordings))
    supervisions = SupervisionSet.from_segments(segments)

    if verbose:
        print(f"Saving {detector_kls.name!r} results ...")
    if output_supervisions_manifest:
        supervisions.to_file(str(output_supervisions_manifest))
    return supervisions
