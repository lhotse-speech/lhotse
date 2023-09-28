import sys
from dataclasses import dataclass
from functools import partial
from io import BytesIO
from pathlib import Path
from time import time
from typing import Any, Callable, Iterable, List, Optional, Tuple, TypeVar, Union

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

if sys.version_info >= (3, 8):
    from typing import Protocol as _Protocol
else:
    _Protocol = object

from typing import Protocol as _Protocol

Detector = Callable[[Recording], IntervalTree]


class Segmental(_Protocol):
    @property
    def start(self) -> float:
        pass

    @start.setter
    def start(self, value: float) -> None:
        pass

    @property
    def duration(self) -> float:
        pass

    @duration.setter
    def duration(self, value: float) -> None:
        pass


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


def to_activity_tree(activities: Iterable[Activity]) -> IntervalTree:
    tree = IntervalTree()
    for activity in activities:
        tree.addi(  # type: ignore
            begin=activity.start,
            end=activity.start + activity.duration,
        )
    tree.merge_overlaps()  # type: ignore
    return tree


def to_trimming_tree(activities: IntervalTree) -> TrimmingTree:
    tree = TrimmingTree(0.0)
    for interval in activities:  # type: ignore
        tree.protect_interval(interval.begin, interval.end)  # type: ignore
    return tree


def make_activity_detector(device: str) -> Detector:
    detector = SileroVAD16k(device=device)
    prepare = partial(to_mono, sampling_rate=detector.sampling_rate)
    # TODO: Need to normalise the sound before analysis?

    def get_detector() -> ActivityDetector:
        return detector  # TODO: Get detector from current scope

    def detect_activity(recording: Recording) -> IntervalTree:
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
    activity_tree: IntervalTree,
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


SegmentalT = TypeVar("SegmentalT", bound=Segmental)


def trim_segmental(
    obj: SegmentalT,
    trimming_tree: TrimmingTree,
    **kwargs: Any,
) -> SegmentalT:
    start = trimming_tree(obj.start)
    duration = trimming_tree(obj.start + obj.duration) - start
    return fastcopy(
        obj,
        start=round(start, ndigits=8),
        duration=round(duration, ndigits=8),
        **kwargs,
    )


def trim_supervision_segment(
    segment: SupervisionSegment,
    trimming_tree: TrimmingTree,
) -> SupervisionSegment:
    # keep the additional supervision information (e.g., speaker, language, etc.)
    new_segment = trim_segmental(
        obj=segment,
        trimming_tree=trimming_tree,
        alignment=None,
    )

    # transform alignment according to the trimming tree
    if segment.alignment is not None:
        trimm = partial(trim_segmental, trimming_tree=trimming_tree)
        new_segment.alignment = {
            key: list(map(trimm, alignment_items))
            for key, alignment_items in segment.alignment.items()
        }

    return new_segment


def trim_supervisions(
    supervisions: Iterable[SupervisionSegment],
    activity_tree: IntervalTree,
) -> List[SupervisionSegment]:
    trimming_tree = to_trimming_tree(activity_tree)
    supervisions = map(
        partial(trim_supervision_segment, trimming_tree=trimming_tree),
        supervisions,
    )
    supervisions = (segment for segment in supervisions if segment.duration > 1e-8)

    # drop supervisions that are have zero duration
    return list(supervisions)


CutT = TypeVar("CutT", bound=Cut)


# TODO: Make sure that the method works with non-data cuts
def trim_cut_by_detector(cut: CutT, detector: Detector) -> CutT:
    # analyse the recording and get the activity tree
    recording = cut.recording
    if recording is None:
        msg = f"Cut {cut.id} does not have a recording"
        raise ValueError(msg)
    if not isinstance(recording, Recording):
        rec_type = type(recording)
        rec_type_repr = f"{rec_type.__module__}.{rec_type.__name__}"
        msg = f"Cut {cut.id!r} has an invalid recording type: {rec_type_repr!r}"
        raise ValueError(msg)
    activity_tree = detector(recording)

    # trim and redefine the recording
    recording = trim_recording(
        recording=recording,
        activity_tree=activity_tree,
    )

    # trim and redefine the supervisions
    supervisions = trim_supervisions(
        supervisions=cut.supervisions,
        activity_tree=activity_tree,
    )
    # TODO: Drop supervisions that are not part of the new audio

    # copy the cut, but replace the recording and supervisions
    return fastcopy(
        cut,
        recording=recording,
        supervisions=supervisions,
        duration=recording.duration,
    )


def trim_mixed_cut(cut: MixedCut, detector: Detector) -> MixedCut:
    raise NotImplementedError("Trimming of mixed cuts is not implemented yet.")


def trim_mono_cut(cut: MonoCut, detector: Detector) -> MonoCut:
    return trim_cut_by_detector(cut=cut, detector=detector)


def trim_multi_cut(cut: MultiCut, detector: Detector) -> MultiCut:
    raise NotImplementedError("Trimming of multi cuts is not implemented yet.")


def trim_data_cut(cut: DataCut, detector: Detector) -> DataCut:
    if isinstance(cut, MonoCut):
        return trim_mono_cut(cut, detector=detector)
    if isinstance(cut, MultiCut):
        return trim_multi_cut(cut, detector=detector)
    raise NotImplementedError("Trimming of this data cut is not implemented yet.")


def trim_padding_cut(cut: PaddingCut, detector: Detector) -> PaddingCut:
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
    detector: Detector,
) -> Tuple[Optional[Cut], TrimmingDetails]:
    start_triming_time = time()
    try:
        try:
            if isinstance(cut, MixedCut):
                trim = trim_mixed_cut(cut=cut, detector=detector)
            elif isinstance(cut, DataCut):
                trim = trim_data_cut(cut=cut, detector=detector)
            elif isinstance(cut, PaddingCut):
                trim = trim_padding_cut(cut=cut, detector=detector)
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


def flush_cut_to_disc(cut: Cut, root: Path) -> Cut:
    # TODO: 1.4 Save recording to root
    return cut


def trim_cut_and_save(
    cut: Cut,
    *,
    root: Optional[Path],
    detector: Detector,
    memorise: bool,
) -> Tuple[Optional[Cut], TrimmingDetails]:
    trim, details = trim_cut(cut=cut, detector=detector)

    if trim is not None and root is not None:
        flush = flush_cut_to_disc(cut=trim, root=root)
        if not memorise:
            trim = flush

    return trim, details


class TrimmingException(ValueError):
    def __init__(self, details: TrimmingDetails):
        self.details = details
        reason = f"{details.reason}" if details.reason else "Unknown error."
        msg = f"Failed to trim cut {details.cut_id!r}. Reason: {reason}"
        super().__init__(msg)


def speach_only(
    cutset: Iterable[Cut],
    root: Optional[Union[str, Path]],
    *,
    skip_exceptions: bool = False,
    device: str = "cpu",
    num_jobs: int = 1,
    verbose: bool = False,
    memorise: bool = False,
    # TODO: save_report: bool = False,
    # TODO: save_recordings_manifest: bool = True,
    # TODO: save_supervisions_manifest: bool = True,
    # TODO: inmemory: bool = True,
) -> Tuple[CutSet, List[TrimmingDetails]]:
    detect_activity: Detector = make_activity_detector(device=device)

    if root is not None:
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

            cut, details = trim_cut_and_save(
                original,
                root=root,
                detector=detect_activity,
                memorise=memorise,
            )
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
