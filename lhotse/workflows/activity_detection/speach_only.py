from dataclasses import dataclass
from functools import partial
from io import BytesIO
from pathlib import Path
from time import time
from typing import Callable, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from intervaltree import IntervalTree  # type: ignore

from lhotse.audio import Recording
from lhotse.audio.backend import torchaudio_save_flac_safe
from lhotse.audio.source import AudioSource
from lhotse.cut import Cut, CutSet
from lhotse.cut.data import DataCut
from lhotse.cut.mixed import MixedCut
from lhotse.cut.mono import MonoCut
from lhotse.cut.multi import MultiCut
from lhotse.cut.padding import PaddingCut
from lhotse.supervision import SupervisionSegment
from lhotse.utils import fastcopy

from .base import Activity, ActivityDetector
from .silero_vad import SileroVAD16k

ActivityTree = IntervalTree
SpeachDetector = Callable[[Recording], ActivityTree]


class TrimmingTree:
    def __init__(self, anchor: float = 0.0):
        self._anchor = anchor
        self._tree = IntervalTree()
        self._tree.addi(-float("inf"), float("inf"))  # type: ignore
        self._tree.slice(self._anchor)  # type: ignore

    def protect_interval(self, start: float, end: float) -> None:
        self._tree.slice(start)  # type: ignore
        self._tree.slice(end)  # type: ignore
        for overlap in self._tree.overlap(start, end):  # type: ignore
            self._tree.remove(overlap)  # type: ignore

    def protect_segment(self, start: float, duration: float) -> None:
        self.protect_interval(start, start + duration)

    def __call__(self, point: float) -> float:
        """Trims a single `point` based on a reference `anchor`"""
        tree = self._tree.copy()
        tree.slice(point)  # type: ignore
        overlap = tree.overlap(*sorted((point, self._anchor)))  # type: ignore
        delta = sum(o.end - o.begin for o in overlap)  # type: ignore
        if point >= self._anchor:
            delta *= -1
        return point + delta

    def __repr__(self):
        """Representation of the trimmer tree"""
        return repr(self._tree).replace("IntervalTree", "TrimmingTree")


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


def to_trimming_tree(activities: ActivityTree) -> TrimmingTree:
    tree = TrimmingTree(0.0)
    for interval in activities:  # type: ignore
        tree.protect_interval(interval.begin, interval.end)  # type: ignore
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


def extract_action_intervals(
    data: np.ndarray,
    sampling_rate: int,
    activity_tree: IntervalTree,
) -> np.ndarray:
    activity_tree = activity_tree.copy()
    activity_tree.merge_overlaps()  # type: ignore
    activity_time_stamps = np.array(sorted(activity_tree))[:, :2].astype(np.float64)
    activity_sample_stamps_raw = activity_time_stamps * sampling_rate
    activity_sample_stamps = activity_sample_stamps_raw.round().astype(np.int64)
    sliced_data = [
        np.take(data, np.arange(start, end), axis=-1)
        for start, end in activity_sample_stamps
    ]
    return np.concatenate(sliced_data, axis=-1)


def trim_recording(
    recording: Recording,
    activity_tree: ActivityTree,
) -> Recording:
    if recording.has_video:  # pragma: no cover
        msg = "Trimming of video recordings is not implemented yet."
        raise NotImplementedError(msg)

    audio = recording.load_audio()

    # keep the original sampling rate
    sampling_rate = recording.sampling_rate

    # keep the original number of channels
    channel_ids = recording.channel_ids
    if isinstance(channel_ids, int):
        channel_ids = [channel_ids]
    channel_ids = tuple(channel_ids)

    # transform audio by removing silence according to activity tree
    trimmed = extract_action_intervals(
        data=audio,
        sampling_rate=sampling_rate,
        activity_tree=activity_tree,
    )
    num_samples = trimmed.shape[-1]
    duration = num_samples / sampling_rate

    # convert to flac and save to memory
    stream = BytesIO()
    torchaudio_save_flac_safe(
        dest=stream,
        src=torch.from_numpy(trimmed),
        sample_rate=sampling_rate,
        format="flac",
    )

    # make new recording storing the trimmed audio
    memory_sources = [
        AudioSource(
            type="memory",
            channels=channel_ids,
            source=stream.getvalue(),
        )
    ]
    return Recording(
        id=recording.id,
        sources=memory_sources,
        sampling_rate=sampling_rate,
        num_samples=num_samples,
        duration=duration,
    )


def trim_supervision_segment(
    segment: SupervisionSegment,
    trimming_tree: TrimmingTree,
) -> SupervisionSegment:
    # keep the additional supervision information (e.g., speaker, language, etc.)
    segment = fastcopy(segment)
    end = segment.end

    # transform supervision according to the trimming tree
    segment.start = round(trimming_tree(segment.start), ndigits=8)
    segment.duration = round(trimming_tree(end) - segment.start, ndigits=8)
    if segment.alignment is not None:
        msg = "Trimming of supervision with alignment is not implemented yet."
        raise NotImplementedError(msg)
    return segment


def trim_supervisions(
    supervisions: Iterable[SupervisionSegment],
    activity_tree: ActivityTree,
) -> List[SupervisionSegment]:
    # TODO: * Drop supervisions that are not part of the new audio
    trimming_tree = to_trimming_tree(activity_tree)
    return [
        trim_supervision_segment(segment=segment, trimming_tree=trimming_tree)
        for segment in supervisions
    ]


def trim_mixed_cut(cut: MixedCut, *, root: Path, detector: SpeachDetector) -> MixedCut:
    raise NotImplementedError("Trimming of mixed cuts is not implemented yet.")


def trim_mono_cut(cut: MonoCut, *, root: Path, detector: SpeachDetector) -> MonoCut:
    # TODO: deepcopy the cut and replace the recording

    # redefine Recording
    recording = cut.recording
    if recording is None:
        message = f"Cut {cut.id} does not have a recording"
        raise ValueError(message)
    activity_tree = detector(recording)
    recording = trim_recording(
        recording=recording,
        activity_tree=activity_tree,
        root=root,
    )
    # TODO: 1.4 Save new recording to root

    # redefine Supervision
    supervisions = trim_supervisions(
        supervisions=cut.supervisions,
        activity_tree=activity_tree,
    )

    # TODO: redefine Cut
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
