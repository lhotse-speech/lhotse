from itertools import chain
from typing import List, Optional

from lhotse.audio import Recording
from lhotse.supervision import SupervisionSegment, SupervisionSet

from .base import Processor, ProcessWorker
from .tools import PathLike, assert_output_file


def _detect_acitvity(recording: Recording, model) -> List[SupervisionSegment]:
    return model(recording)


def detect_activity(
    recordings: Recording,
    detector_kls,
    model_name: str,
    # output
    output_supervisions_manifest: Optional[PathLike] = None,
    # mode
    num_jobs: int = 1,
    verbose: bool = False,
) -> SupervisionSet:
    output_supervisions_manifest = assert_output_file(
        output_supervisions_manifest, "output_supervisions_manifest"
    )

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

    if verbose:
        print(f"Saving {model_name!r} results ...")
    if output_supervisions_manifest:
        supervisions.to_file(str(output_supervisions_manifest))
    return supervisions
