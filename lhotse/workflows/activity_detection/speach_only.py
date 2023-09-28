__all__ = (
    "TrimmingTree",
    "make_activity_detector",
    "trim_recording",
    "trim_segmental",
    "trim_supervision_segment",
    "trim_supervisions",
    "TrimmingDetails",
    "trim_cut",
    "move_recording_to_disc",
    "trim_cut_and_save",
    "TrimmingException",
    "speach_only",
)
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
from torch import Tensor
from torchaudio import save as torchaudio_save  # type: ignore

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

import warnings
from typing import Protocol as _Protocol

Detector = Callable[[Recording], IntervalTree]
PathLike = Union[str, Path]


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
        self.anchor = anchor
        self._tree = IntervalTree()
        self._tree.addi(-float("inf"), float("inf"))  # type: ignore
        self._tree.slice(self.anchor)  # type: ignore

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
        overlap = tree.overlap(*sorted((point, self.anchor)))  # type: ignore
        delta = sum(o.end - o.begin for o in overlap)  # type: ignore
        if point >= self.anchor:
            delta *= -1
        return point + delta

    def __repr__(self):
        """Representation of the trimmer tree"""
        return repr(self._tree).replace("IntervalTree", "TrimmingTree")


def _to_mono(recording: Recording, *, sampling_rate: int) -> Recording:
    """Converts a recording to mono and resamples it to the given sampling rate"""
    mono = recording  # FIXME: Convert the recording to mono
    resampled = mono.resample(sampling_rate)
    return resampled


def _to_activity_tree(activities: Iterable[Activity]) -> IntervalTree:
    tree = IntervalTree()
    for activity in activities:
        tree.addi(  # type: ignore
            begin=activity.start,
            end=activity.start + activity.duration,
        )
    tree.merge_overlaps()  # type: ignore
    return tree


def to_trimming_tree(
    activities: IntervalTree,
    anchor: float = 0.0,
    duration: Optional[float] = None,
    protect_outside: bool = False,
) -> TrimmingTree:
    tree = TrimmingTree(anchor)
    for interval in activities:  # type: ignore
        tree.protect_interval(interval.begin, interval.end)  # type: ignore

    if protect_outside:
        tree.protect_interval(-float("inf"), anchor)
        if duration is not None:
            tree.protect_interval(anchor + duration, float("inf"))
    return tree


def make_activity_detector(device: str) -> Detector:
    detector = SileroVAD16k(device=device)
    prepare = partial(_to_mono, sampling_rate=detector.sampling_rate)
    # TODO: Need to normalise the sound before analysis?

    def get_detector() -> ActivityDetector:
        return detector  # TODO: Get detector from current scope

    def detect_activity(recording: Recording) -> IntervalTree:
        """Detects activity timestamps in a recording"""
        detector = get_detector()
        track = prepare(recording).load_audio()  # type: ignore
        activities = detector.forward(track)  # type: ignore
        return _to_activity_tree(activities)

    return detect_activity


def _extract_action_intervals(
    data: np.ndarray,
    *,
    sampling_rate: int,
    activities: IntervalTree,
) -> np.ndarray:
    activities = activities.copy()
    activities.merge_overlaps()  # type: ignore
    activity_time_stamps = np.array(sorted(activities))[:, :2].astype(np.float64)
    activity_sample_stamps_raw = activity_time_stamps * sampling_rate
    activity_sample_stamps = activity_sample_stamps_raw.round().astype(np.int64)
    sliced_data = [
        np.take(data, np.arange(start, end), axis=-1)
        for start, end in activity_sample_stamps
    ]
    return np.concatenate(sliced_data, axis=-1)


def trim_recording(recording: Recording, activities: IntervalTree) -> Recording:
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
    trimmed = _extract_action_intervals(
        data=audio,
        sampling_rate=sampling_rate,
        activities=activities,
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


def trim_segmental(obj: SegmentalT, trimmer: TrimmingTree, **kwargs: Any) -> SegmentalT:
    start = trimmer(obj.start)
    duration_ = trimmer(obj.start + obj.duration) - start
    return fastcopy(
        obj,
        start=round(start, ndigits=8),
        duration=round(duration_, ndigits=8),
        **kwargs,
    )


def trim_supervision_segment(
    segment: SupervisionSegment,
    trimmer: TrimmingTree,
) -> SupervisionSegment:
    # keep the additional supervision information (e.g., speaker, language, etc.)
    new_segment = trim_segmental(
        obj=segment,
        trimmer=trimmer,
        alignment=None,
    )

    # transform alignment according to the trimming tree
    if segment.alignment is not None:
        trimm = partial(trim_segmental, trimmer=trimmer)
        new_segment.alignment = {
            key: list(map(trimm, alignment_items))
            for key, alignment_items in segment.alignment.items()
        }

    return new_segment


def trim_supervisions(
    supervisions: Iterable[SupervisionSegment],
    trimmer: TrimmingTree,
) -> List[SupervisionSegment]:
    supervisions = map(
        partial(trim_supervision_segment, trimmer=trimmer),
        supervisions,
    )

    # drop supervisions that are have zero duration
    supervisions = (segment for segment in supervisions if segment.duration > 1e-8)
    return list(supervisions)


CutT = TypeVar("CutT", bound=Cut)


# TODO: Make sure that the method works with non-data cuts
def _trim_cut(cut: CutT, *, detector: Detector, protect_outside: bool) -> CutT:
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
        activities=activity_tree,
    )

    trimming_tree = to_trimming_tree(
        activity_tree,
        anchor=0.0,
        # if supervisions have a duration that exceeds the recording duration, we need to protect it
        duration=recording.duration,
        # if supervisions have a negative start, we need to protect it
        protect_outside=protect_outside,
    )

    # trim and redefine the supervisions
    supervisions = trim_supervisions(
        supervisions=cut.supervisions,
        trimmer=trimming_tree,
    )
    # TODO: Drop supervisions that are not part of the new audio

    # copy the cut, but replace the recording and supervisions
    return fastcopy(
        cut,
        recording=recording,
        supervisions=supervisions,
        duration=recording.duration,
    )


def _trim_mixed_cut(
    cut: MixedCut, *, detector: Detector, protect_outside: bool
) -> MixedCut:
    raise NotImplementedError("Trimming of mixed cuts is not implemented yet.")


def _trim_mono_cut(
    cut: MonoCut, *, detector: Detector, protect_outside: bool
) -> MonoCut:
    return _trim_cut(cut=cut, detector=detector, protect_outside=protect_outside)


def _trim_multi_cut(
    cut: MultiCut, *, detector: Detector, protect_outside: bool
) -> MultiCut:
    raise NotImplementedError("Trimming of multi cuts is not implemented yet.")


def _trim_data_cut(
    cut: DataCut, *, detector: Detector, protect_outside: bool
) -> DataCut:
    if isinstance(cut, MonoCut):
        return _trim_mono_cut(
            cut=cut, detector=detector, protect_outside=protect_outside
        )
    if isinstance(cut, MultiCut):
        return _trim_multi_cut(
            cut=cut, detector=detector, protect_outside=protect_outside
        )
    raise NotImplementedError("Trimming of this data cut is not implemented yet.")


def _trim_padding_cut(
    cut: PaddingCut, *, detector: Detector, protect_outside: bool
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
    detector: Detector,
    protect_outside: bool = True,
) -> Tuple[Optional[Cut], TrimmingDetails]:
    cut = cut.drop_features()
    if not isinstance(cut, Cut):  # type: ignore
        details = TrimmingDetails(
            cut_id=getattr(cut, "id", "unknown"),
            error=True,
            reason=f"Cut is not an instance of {Cut.__name__!r}",
            elapsed_time=0.0,
        )
        return None, details

    start_triming_time = time()
    try:
        try:
            if isinstance(cut, MixedCut):
                trim = _trim_mixed_cut(
                    cut, detector=detector, protect_outside=protect_outside
                )
            elif isinstance(cut, DataCut):
                trim = _trim_data_cut(
                    cut, detector=detector, protect_outside=protect_outside
                )
            elif isinstance(cut, PaddingCut):
                trim = _trim_padding_cut(
                    cut, detector=detector, protect_outside=protect_outside
                )
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


def move_recording_to_disc(
    recording: Recording,
    root: Union[str, Path],
    extension: str = "flac",
    absolute: bool = False,
) -> Recording:
    # save recording to root

    if recording.has_video:  # pragma: no cover
        msg = "Saving of video recordings is not implemented yet."
        raise NotImplementedError(msg)
    root = Path(root).expanduser().resolve().absolute()
    if not root.is_dir():
        raise ValueError(f"Saving root '{root}' is not a directory.")
    path = root / f"{recording.id}.{extension}"
    if path.exists():
        raise ValueError(f"File {path} already exists.")

    try:
        # if ext == "flac":
        #     recording = recording.move_to_memory(format=ext)
        #     data = recording.sources[0].source
        #     if not isinstance(data, bytes):
        #         raise ValueError(f"Recording {recording.id} has invalid data type.")
        #     with path.open("wb") as file:
        #         file.write(data)
        torchaudio_save(
            path,
            Tensor(recording.load_audio()),
            sample_rate=recording.sampling_rate,
            format=extension,
            channels_first=True,
        )

        # use relative path to the root
        source = AudioSource(
            type="file",
            channels=recording.channel_ids,
            source=str(path if absolute else path.relative_to(Path.cwd())),
        )
        return Recording(
            id=recording.id,
            sources=[source],
            sampling_rate=recording.sampling_rate,
            num_samples=recording.num_samples,
            duration=recording.duration,
        )

    except Exception as exc:
        if path.exists():
            path.unlink()
        raise exc


def trim_cut_and_save(
    cut: Cut,
    *,
    storage_dir: Optional[Path],
    detector: Detector,
    memorise_recording: bool,
    save_with_extension: str = "flac",
    use_absolute_path: bool = False,
    protect_outside: bool = True,
) -> Tuple[Optional[Cut], TrimmingDetails]:
    trim, details = trim_cut(
        cut=cut,
        detector=detector,
        protect_outside=protect_outside,
    )

    if trim is not None and storage_dir is not None:
        # FIXME: trim may not have a recording
        recording = move_recording_to_disc(
            recording=trim.recording,
            root=storage_dir,
            extension=save_with_extension,
            absolute=use_absolute_path,
        )
        if not memorise_recording:
            trim.recording = recording

    return trim, details


class TrimmingException(ValueError):
    def __init__(self, details: TrimmingDetails):
        self.details = details
        reason = f"{details.reason}" if details.reason else "Unknown error."
        msg = f"Failed to trim cut {details.cut_id!r}. Reason: {reason}"
        super().__init__(msg)


def _resolve_path(path: Optional[PathLike]) -> Optional[Path]:
    if path is None or (isinstance(path, str) and path == ""):
        return None
    return Path(path).expanduser().resolve().absolute()


def _assert_output_dir(path: Optional[PathLike], name: str) -> Optional[Path]:
    path = _resolve_path(path)
    if path is None:
        return None
    if path.is_file():
        msg = f"Path {name}={path} is a file."
        msg += " Please provide a directory path."
        raise ValueError(msg)
    if not path.exists():
        msg = f"Directory {name}={path} does not exist."
        msg += " Please create {path} first or provide a different path."
        raise ValueError(msg)

    return path


def speach_only(
    cutset: Iterable[Cut],
    *,
    # output
    keep_in_memory: bool = False,
    output_dir: Optional[PathLike] = None,
    output_recordings_extension: str = "flac",
    # options
    use_absolute_paths: bool = False,
    protect_outside: bool = True,
    skip_exceptions: bool = False,
    # mode
    device: str = "cpu",
    num_jobs: int = 1,
    verbose: bool = False,
) -> Tuple[CutSet, List[TrimmingDetails]]:
    output_dir = _assert_output_dir(output_dir, "output_dir")

    if not (keep_in_memory or output_dir):
        msg = "If the recordings are not kept in memory, they must be saved to disk."
        msg += " Please provide a output_dir or set keep_in_memory=True."
        raise ValueError(msg)

    # if output_report_path is not None:
    #     if output_report_path.suffix == "":
    #         output_report_path = Path(str(output_report_path) + ".csv")
    #     elif output_report_path.suffix not in [".csv", ".json", ".yaml"]:
    #         msg = "Report file must have one of the following extensions: .csv, .json, .yaml"
    #         raise ValueError(msg)

    storage_dir = None
    output_report_path = None
    output_cuts_manifest_path = None

    if output_dir is not None:
        storage_dir = output_dir / "storage"
        storage_dir.mkdir(parents=False, exist_ok=True)
        output_report_path = output_dir / "speach-only-report.csv"
        output_cuts_manifest_path = output_dir / "cuts.json.gz"

    # TODO: * Use multiprocessing to speed up the process
    # TODO: * Balance the load across processes
    # TODO: * Do not use more processes than the number of available CPUs
    # TODO: * Do not use more processes than the number of cuts
    # TODO: * Be careful not to overload the RAM
    # TODO: * Separate the cutset into chunks and process them separately?

    if verbose:
        from tqdm.auto import tqdm  # pylint: disable=C0415

        cutset = tqdm(cutset, desc="Trimming cuts", unit="cut")

    detect_activity: Detector = make_activity_detector(device=device)
    processor = partial(
        trim_cut_and_save,
        storage_dir=storage_dir,
        detector=detect_activity,
        memorise_recording=keep_in_memory,
        save_with_extension=output_recordings_extension,
        use_absolute_path=use_absolute_paths,
        protect_outside=protect_outside,
    )

    cuts: List[Cut] = []
    report: List[TrimmingDetails] = []

    for cut in cutset:
        cut, details = processor(cut)
        report.append(details)
        if cut is not None:
            cuts.append(cut)
        if not skip_exceptions and (details.error or cut is None):
            raise TrimmingException(details)

    result = CutSet.from_cuts(cuts)

    if output_cuts_manifest_path is not None:
        try:
            result.to_file(output_cuts_manifest_path)
        except Exception as exc:
            if output_cuts_manifest_path.exists():
                output_cuts_manifest_path.unlink()
            if verbose:
                print(
                    f"Failed to save cut manifest to {output_cuts_manifest_path}: {exc}"
                )

    if output_dir is not None:
        try:
            result.decompose(str(output_dir), verbose=verbose)
            features_path = output_dir / "features.jsonl.gz"
            if features_path.exists():
                features_path.unlink()
        except Exception as exc:
            if verbose:
                print(f"Failed to save decomposed manifest to {output_dir}: {exc}")

    if output_report_path is not None:
        try:
            import pandas as pd  # pylint: disable=C0415

            pd.DataFrame(report).to_csv(output_report_path, index=False)

        except Exception as exc:
            if output_report_path.exists():
                output_report_path.unlink()
            if verbose:
                print(f"Failed to save report to {output_report_path}: {exc}")

    # return the new cutset and the report of trimming
    return result, report
