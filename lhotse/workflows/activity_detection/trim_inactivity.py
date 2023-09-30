__all__ = (
    "InactivityTrimmer",
    "TrimmingDetails",
    "TrimmingException",
    "trim_inactivity",
)


from dataclasses import dataclass
from functools import partial
from io import BytesIO
from pathlib import Path
from time import time
from typing import Any, Iterable, List, Optional, Tuple, Type, TypeVar, Union

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
from lhotse.workflows.backend import Processor, ProcessWorker

from ._tools import PathLike, assert_output_dir
from .base import Activity, ActivityDetector


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
    """Converts a list of activities to an activity tree"""
    tree = IntervalTree()
    for activity in activities:
        tree.addi(  # type: ignore
            begin=activity.start,
            end=activity.start + activity.duration,
        )
    tree.merge_overlaps()  # type: ignore
    return tree


def _to_trimming_tree(
    activities: IntervalTree,
    anchor: float = 0.0,
    duration: Optional[float] = None,
    protect_outside: bool = False,
) -> TrimmingTree:
    """Converts an activity tree to a trimming tree"""
    tree = TrimmingTree(anchor)
    for interval in activities:  # type: ignore
        tree.protect_interval(interval.begin, interval.end)  # type: ignore

    if protect_outside:
        tree.protect_interval(-float("inf"), anchor)
        if duration is not None:
            tree.protect_interval(anchor + duration, float("inf"))
    return tree


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


SegmentalT = TypeVar("SegmentalT")
CutT = TypeVar("CutT", bound=Cut)


def _trim_segmental(
    obj: SegmentalT,
    trimmer: TrimmingTree,
    **kwargs: Any,
) -> SegmentalT:
    start = trimmer(obj.start)
    duration_ = trimmer(obj.start + obj.duration) - start
    return fastcopy(
        obj,
        start=round(start, ndigits=8),
        duration=round(duration_, ndigits=8),
        **kwargs,
    )


class InactivityTrimmer:
    """Trims a recording and its supervisions based on the activity detector."""

    def __init__(
        self,
        detector: ActivityDetector,
        protect_outside: bool = True,
        storage_dir: Optional[Path] = None,
        save_with_extension: str = "flac",
    ):
        self._detector = detector
        self._protect_outside = protect_outside
        self._storage_dir = storage_dir
        self._extension = save_with_extension

    def __trim_recording(
        self, recording: Recording, activities: IntervalTree
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
        trimmed = _extract_action_intervals(
            data=audio,
            sampling_rate=sampling_rate,
            activities=activities,
        )

        if self._storage_dir is None:
            # convert to flac and save to memory
            stream = BytesIO()
            torchaudio_save_flac_safe(
                dest=stream,
                src=torch.from_numpy(trimmed),
                sample_rate=sampling_rate,
                format="flac",
            )

            # make new recording storing the trimmed audio
            source = AudioSource(
                type="memory",
                channels=channel_ids,
                source=stream.getvalue(),
            )
        else:
            filename = f"{recording.id}.{self._extension}"
            path = self._storage_dir / filename
            if path.exists():
                raise ValueError(f"File {path} already exists")
            try:
                torchaudio_save(
                    path,
                    Tensor(trimmed),
                    sample_rate=recording.sampling_rate,
                    format=self._extension,
                    channels_first=True,
                )
                source = AudioSource(
                    type="file",
                    channels=channel_ids,
                    source=str(Path(path.parent.name) / path.name),
                )
            except Exception as exc:
                if path.exists():
                    path.unlink()
                raise exc

        num_samples = trimmed.shape[-1]
        duration = num_samples / sampling_rate
        return Recording(
            id=recording.id,
            sources=[source],
            sampling_rate=sampling_rate,
            num_samples=num_samples,
            duration=duration,
        )

    def _trim_recording(self, recording: Recording) -> Tuple[Recording, IntervalTree]:
        track = _to_mono(recording, sampling_rate=self._detector.sampling_rate)
        audio = track.load_audio()  # type: ignore
        activities = self._detector(audio)
        activity_tree = _to_activity_tree(activities)
        recording = self.__trim_recording(recording, activity_tree)
        return recording, activity_tree

    def _trim_supervision_segment(
        self,
        segment: SupervisionSegment,
        trimmer: TrimmingTree,
    ) -> SupervisionSegment:
        # keep the additional supervision information (e.g., speaker, language, etc.)
        new_segment = _trim_segmental(
            obj=segment,
            trimmer=trimmer,
            alignment=None,
        )

        # transform alignment according to the trimming tree
        if segment.alignment is not None:
            trimm = partial(_trim_segmental, trimmer=trimmer)
            new_segment.alignment = {
                key: list(map(trimm, alignment_items))
                for key, alignment_items in segment.alignment.items()
            }

        return new_segment

    def _trim_supervisions(
        self,
        supervisions: Iterable[SupervisionSegment],
        trimmer: TrimmingTree,
    ) -> List[SupervisionSegment]:
        trim = partial(self._trim_supervision_segment, trimmer=trimmer)
        supervisions = map(trim, supervisions)

        # drop supervisions that are have zero duration
        supervisions = (segment for segment in supervisions if segment.duration > 1e-8)
        return list(supervisions)

    def trim_recording(self, recording: Recording) -> Recording:
        recording, _ = self._trim_recording(recording)
        return recording

    def trim(
        self,
        recording: Recording,
        supervisions: Optional[Iterable[SupervisionSegment]] = None,
    ) -> Tuple[Recording, Optional[List[SupervisionSegment]]]:
        """Trims a recording and its supervisions based on the activity detector"""
        # trim and redefine the recording
        recording, activity_tree = self._trim_recording(recording)

        # trim and redefine the supervisions
        if supervisions is not None:
            trimming_tree = _to_trimming_tree(
                activity_tree,
                anchor=0.0,
                # if supervisions have a duration that exceeds
                # the recording duration, we need to protect it
                duration=recording.duration,
                # if supervisions have a negative start, we need to protect it
                protect_outside=self._protect_outside,
            )
            supervisions = self._trim_supervisions(
                supervisions=supervisions,
                trimmer=trimming_tree,
            )
            # TODO: Drop supervisions that are not part of the new audio

        return recording, supervisions

    def _trim_mixedcut(self, cut: MixedCut) -> MixedCut:
        raise NotImplementedError("Trimming of MixedCut is not implemented yet")

    def _trim_monocut(self, cut: MonoCut) -> MonoCut:
        recording = cut.recording
        if recording is None:
            raise ValueError("Cannot trim a cut without a recording")
        recording, supervisions = self.trim(
            recording=recording,
            supervisions=cut.supervisions,
        )
        # copy the cut, but replace the recording and supervisions
        return fastcopy(
            cut,
            recording=recording,
            supervisions=supervisions,
            duration=recording.duration,
        )

    def _trim_multicut(self, cut: MultiCut) -> MultiCut:
        raise NotImplementedError("Trimming of MultiCut is not implemented yet")

    def _trim_datacut(self, cut: DataCut) -> DataCut:
        if isinstance(cut, MonoCut):
            return self._trim_monocut(cut=cut)
        if isinstance(cut, MultiCut):
            return self._trim_multicut(cut=cut)
        raise NotImplementedError("Trimming of DataCut is not implemented yet")

    def _trim_paddingcut(self, cut: PaddingCut) -> PaddingCut:
        raise NotImplementedError("Trimming of PaddingCut is not implemented yet")

    def trim_cut(self, cut: CutT) -> CutT:
        """Trims a cut based on the activity detector"""
        cut = cut.drop_features()
        if isinstance(cut, MixedCut):
            return self._trim_mixedcut(cut)
        if isinstance(cut, DataCut):
            return self._trim_datacut(cut)
        if isinstance(cut, PaddingCut):
            return self._trim_paddingcut(cut)
        cut_type = f"{cut.__class__.__module__}.{cut.__class__.__name__}"
        msg = f"Cut has an unsupported type {cut_type!r}"
        raise NotImplementedError(msg)


@dataclass
class TrimmingDetails:
    cut_id: str
    error: bool
    reason: Optional[str]
    elapsed_time: float


def _trim_inactivity(
    cut: Cut,
    *,
    model: ActivityDetector,
    storage_dir: Optional[Path],
    save_with_extension: str,
    protect_outside: bool,
    skip_exceptions: bool,
) -> Tuple[Optional[Cut], TrimmingDetails]:
    start_triming_time = time()
    try:
        trimmer = InactivityTrimmer(
            detector=model,
            storage_dir=storage_dir,
            protect_outside=protect_outside,
            save_with_extension=save_with_extension,
        )
        trim = trimmer.trim_cut(cut)

        completed_in = time() - start_triming_time
        details = TrimmingDetails(
            cut_id=cut.id,
            error=False,
            reason=None,
            elapsed_time=completed_in,
        )
        return trim, details
    except Exception as exc:
        if not skip_exceptions:
            raise exc
        completed_in = time() - start_triming_time
        return None, TrimmingDetails(
            cut_id=cut.id,
            error=True,
            reason=str(exc),
            elapsed_time=completed_in,
        )


class TrimmingException(Exception):
    def __init__(self, details: TrimmingDetails):
        self.details = details
        reason = f"{details.reason}" if details.reason else "Unknown error."
        msg = f"Failed to trim cut {details.cut_id!r}. Reason: {reason}"
        super().__init__(msg)


def trim_inactivity(
    cutset: Iterable[Cut],
    *,
    detector: Union[str, Type[ActivityDetector]],
    # output
    output_dir: Optional[PathLike] = None,
    output_recordings_extension: str = "flac",
    # options
    protect_outside: bool = True,
    # mode
    device: str = "cpu",
    num_jobs: int = 1,
    verbose: bool = False,
    skip_exceptions: bool = False,
    warnings_mode: Optional[str] = None,
) -> Tuple[CutSet, List[TrimmingDetails]]:
    output_dir = assert_output_dir(output_dir, "output_dir")

    if output_dir is None:
        if output_recordings_extension != "flac":
            msg = "When keeping recordings in memory, only flac is supported."
            msg += f" Please provide output_dir to save recordings as {output_recordings_extension}."
            raise ValueError(msg)

    storage_dir = None
    output_report_path = None
    output_cuts_manifest_path = None

    if output_dir is not None:
        storage_dir = output_dir / "storage"
        storage_dir.mkdir(parents=False, exist_ok=True)
        output_report_path = output_dir / "speach-only-report.csv"
        output_cuts_manifest_path = output_dir / "cuts.json.gz"

    options = {
        "storage_dir": storage_dir,
        "save_with_extension": output_recordings_extension,
        "protect_outside": protect_outside,
        "skip_exceptions": skip_exceptions,
    }

    detector_ = ActivityDetector.get_detector(detector)

    worker = ProcessWorker(
        gen_model=partial(detector_, device=device),
        do_work=_trim_inactivity,
        warnings_mode=warnings_mode,
    )

    processor = Processor(worker, num_jobs=num_jobs, verbose=verbose)

    cuts: List[Cut] = []
    report: List[TrimmingDetails] = []

    for cut, details in processor(cutset, **options):
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
                path = output_cuts_manifest_path
                print(f"Failed to save cut manifest to {path}: {exc}")

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
    if output_dir is not None:
        result = result.with_recording_path_prefix(str(output_dir))
    return result, report
