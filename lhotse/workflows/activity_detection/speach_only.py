from dataclasses import dataclass
from functools import partial
from pathlib import Path
from time import time
from typing import Callable, Iterable, List, Optional, Tuple, Union

from intervaltree import IntervalTree  # type: ignore

from lhotse.audio import Recording
from lhotse.cut import Cut, CutSet
from lhotse.cut.data import DataCut
from lhotse.cut.mixed import MixedCut
from lhotse.cut.mono import MonoCut
from lhotse.cut.multi import MultiCut
from lhotse.cut.padding import PaddingCut
from lhotse.supervision import SupervisionSegment

from .base import Activity, ActivityDetector
from .silero_vad import SileroVAD16k

ActivityTree = IntervalTree
SpeachDetector = Callable[[Recording], ActivityTree]


def to_mono(recording: Recording, sampling_rate: int) -> Recording:
    """Converts a recording to mono and resamples it to the given sampling rate"""
    mono = recording  # FIXME: Convert the recording to mono
    resampled = mono.resample(sampling_rate)
    return resampled


def to_activity_tree(activities: Iterable[Activity]) -> ActivityTree:
    tree = IntervalTree()
    for activity in activities:
        tree.addi(  # type: ignore
            begin=activity.start,
            end=activity.start + activity.duration,
        )
    tree.merge_overlaps()  # type: ignore
    return tree


def make_activity_detector(device: str) -> SpeachDetector:
    detector = SileroVAD16k(device=device)
    prepare = partial(to_mono, sampling_rate=detector.sampling_rate)
    # TODO: Need to normalise the sound before analysis?

    def get_detector() -> ActivityDetector:
        return detector  # TODO: Get detector from current scope

    def detect_activity(recording: Recording) -> ActivityTree:
        """Detects activity timestamps in a recording"""
        detector = get_detector()
        track = prepare(recording).load_audio()  # type: ignore
        activities = detector.forward(track)  # type: ignore
        return to_activity_tree(activities)

    return detect_activity


def trim_recording(recording: Recording, tree: ActivityTree, root: Path) -> Recording:
    # TODO: 1.3 Transform audio by removing silence according to selected fragments
    # TODO: * Keep the original number of channels
    # TODO: * Keep the original sampling rate

    # TODO: 1.4 Save new audio to root
    # TODO: * Make sure that the new audio is saved in the same format as the original audio
    raise NotImplementedError("Trimming of recordings is not implemented yet.")


def trim_supervision_segment(
    segment: SupervisionSegment, tree: ActivityTree
) -> SupervisionSegment:
    # TODO: 1.5 Transform supervision according to selected fragments
    # TODO: * Keep the additional supervision information (e.g., speaker, language, etc.)
    raise NotImplementedError(
        "Trimming of supervision segments is not implemented yet."
    )


def trim_supervisions(
    supervisions: Iterable[SupervisionSegment],
    tree: ActivityTree,
) -> List[SupervisionSegment]:
    # TODO: * Drop supervisions that are not part of the new audio
    raise NotImplementedError("Trimming of supervisions is not implemented yet.")


def trim_mixed_cut(cut: MixedCut, *, root: Path, detector: SpeachDetector) -> MixedCut:
    raise NotImplementedError("Trimming of mixed cuts is not implemented yet.")


def trim_mono_cut(cut: MonoCut, *, root: Path, detector: SpeachDetector) -> MonoCut:
    recording = cut.recording
    if recording is None:
        message = f"Cut {cut.id} does not have a recording"
        raise ValueError(message)
    activity_tree = detector(recording)

    # Redefine Recording
    trimmed = trim_recording(recording, activity_tree, root)

    # TODO: deepcopy the cut and replace the recording
    raise NotImplementedError("Trimming of mono cuts is not implemented yet.")


def trim_multi_cut(cut: MultiCut, *, root: Path, detector: SpeachDetector) -> MultiCut:
    raise NotImplementedError("Trimming of multi cuts is not implemented yet.")


def trim_data_cut(cut: DataCut, *, root: Path, detector: SpeachDetector) -> DataCut:
    if isinstance(cut, MonoCut):
        return trim_mono_cut(cut, root=root, detector=detector)
    if isinstance(cut, MultiCut):
        return trim_multi_cut(cut, root=root, detector=detector)
    raise NotImplementedError("Trimming of this data cut is not implemented yet.")


def trim_padding_cut(
    cut: PaddingCut, *, root: Path, detector: SpeachDetector
) -> PaddingCut:
    raise NotImplementedError("Trimming of padding cuts is not implemented yet.")


@dataclass
class TrimmingDetails:
    cut_id: str
    error: bool
    reason: Optional[str]
    elapsed_time: float


def trim_cut(
    cut: Cut,
    *,
    root: Path,
    detector: SpeachDetector,
) -> Tuple[Optional[Cut], TrimmingDetails]:
    start_triming_time = time()
    try:
        try:
            if isinstance(cut, MixedCut):
                trim = trim_mixed_cut(cut=cut, root=root, detector=detector)
            elif isinstance(cut, DataCut):
                trim = trim_data_cut(cut=cut, root=root, detector=detector)
            elif isinstance(cut, PaddingCut):
                trim = trim_padding_cut(cut=cut, root=root, detector=detector)
            else:
                raise NotImplementedError()

            completed_in = time() - start_triming_time
            details = TrimmingDetails(
                cut_id=cut.id,
                error=False,
                reason=None,
                elapsed_time=completed_in,
            )
            return trim, details

        except NotImplementedError as exc:
            cut_type = f"{cut.__class__.__module__}.{cut.__class__.__name__}"
            exc_string = f" {exc}" if str(exc) != "" else ""
            msg = f"Cut has an unsupported type {cut_type!r}.{exc_string}"
            raise NotImplementedError(msg) from exc

    except Exception as exc:
        completed_in = time() - start_triming_time
        return None, TrimmingDetails(
            cut_id=cut.id,
            error=True,
            reason=str(exc),
            elapsed_time=completed_in,
        )


class TrimmingException(ValueError):
    def __init__(self, details: TrimmingDetails):
        self.details = details
        reason = f"{details.reason}" if details.reason else "Unknown error."
        msg = f"Failed to trim cut {details.cut_id!r}. Reason: {reason}"
        super().__init__(msg)


def speach_only(
    cutset: Iterable[Cut],
    root: Union[str, Path],
    *,
    skip_exceptions: bool = False,
    device: str = "cpu",
    num_jobs: int = 1,
    verbose: bool = False,
    # TODO: save_report: bool = False,
    # TODO: save_recordings_manifest: bool = True,
    # TODO: save_supervisions_manifest: bool = True,
    # TODO: inmemory: bool = True,
) -> Tuple[CutSet, List[TrimmingDetails]]:
    detect_activity: SpeachDetector = make_activity_detector(device=device)
    root = Path(root).expanduser().resolve().absolute()

    if not root.is_dir():
        raise ValueError(f"Saving root '{root}' is not a directory.")
    if not root.exists():
        raise ValueError(f"Saving root '{root}' does not exist.")

    cuts: List[Cut] = []
    report: List[TrimmingDetails] = []

    # TODO: * Use multiprocessing to speed up the process
    # TODO: * Balance the load across processes
    # TODO: * Do not use more processes than the number of available CPUs
    # TODO: * Do not use more processes than the number of cuts
    # TODO: * Be careful not to overload the RAM
    # TODO: * Separate the cutset into chunks and process them separately?
    # TODO: * Use tqdm to show progress

    if verbose:
        from tqdm.auto import tqdm  # pylint: disable=C0415

        cutset = tqdm(cutset, desc="Trimming cuts", unit="cut")

    for i, original in enumerate(cutset):
        try:
            if not isinstance(original, Cut):  # type: ignore
                details = TrimmingDetails(
                    cut_id=getattr(original, "id", None) or f"cut-{i}",
                    error=True,
                    reason=f"Cutset contains an object that is not a Cut: {original}",
                    elapsed_time=0.0,
                )
                raise TrimmingException(details)

            cut, details = trim_cut(original, root=root, detector=detect_activity)
            if cut is None or details.error:
                raise TrimmingException(details)

            cuts.append(cut)

        except TrimmingException as exc:
            report.append(exc.details)
            if skip_exceptions:
                continue
            raise exc

    # return the new cutset and the report of trimming
    return CutSet.from_cuts(cuts), report
