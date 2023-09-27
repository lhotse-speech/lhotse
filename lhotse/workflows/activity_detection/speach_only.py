from functools import partial
from typing import Callable, Iterable

from intervaltree import IntervalTree

from lhotse.audio import Recording
from lhotse.cut import CutSet

from .base import Activity, ActivityDetector
from .silero_vad import SileroVAD16k


def convert_recording_to_mono(recording: Recording, sampling_rate: int) -> Recording:
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


def make_activity_detector(device: str) -> Callable[[Recording], IntervalTree]:
    detector = SileroVAD16k(device=device)
    prepare = partial(convert_recording_to_mono, sampling_rate=detector.sampling_rate)
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


def speach_only(
    cutset: CutSet,
    root: str,
    *,
    device: str = "cpu",
    num_jobs: int = 1,
) -> CutSet:
    # TODO: 1. Act on cutset elements, for each cut:
    detect_activity = make_activity_detector(device=device)
    # TODO: 1.3 Transform audio by removing silence according to selected fragments
    # TODO: 1.4 Save new audio to root
    # TODO: 1.4 Redefine Recording
    # TODO: 1.5 Transform supervision according to selected fragments
    # TODO: 1.7 Form a new cutset element
    # TODO: 2. Collect and return a new cutset
    # TODO: * Use multiprocessing to speed up the process
    # TODO: * Balance the load across processes
    # TODO: * Do not use more processes than the number of available CPUs
    # TODO: * Do not use more processes than the number of cuts
    # TODO: * Be careful not to overload the RAM
    # TODO: * Separate the cutset into chunks and process them separately?
    # TODO: * Save the new audio to disk during processing
    # TODO: * Make sure that the new audio is saved in the same format as the original audio
    # TODO: * Keep the original number of channels
    # TODO: * Keep the original sampling rate
    # TODO: * Drop supervisions that are not part of the new audio
    # TODO: * Keep the additional supervision information (e.g., speaker, language, etc.)
    # TODO: * Use tqdm to show progress
    raise NotImplementedError()
