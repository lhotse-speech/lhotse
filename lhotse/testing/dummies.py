from typing import Type

from lhotse.audio import Recording, RecordingSet
from lhotse.cut import Cut, CutSet
from lhotse.features import FeatureSet, Features
from lhotse.manipulation import Manifest
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import fastcopy


# noinspection PyPep8Naming
def DummyManifest(type_: Type, *, begin_id: int, end_id: int) -> Manifest:
    if type_ == RecordingSet:
        return RecordingSet.from_recordings(dummy_recording(idx) for idx in range(begin_id, end_id))
    if type_ == SupervisionSet:
        return SupervisionSet.from_segments(dummy_supervision(idx) for idx in range(begin_id, end_id))
    if type_ == FeatureSet:
        # noinspection PyTypeChecker
        return FeatureSet.from_features(dummy_features(idx) for idx in range(begin_id, end_id))
    if type_ == CutSet:
        # noinspection PyTypeChecker
        return CutSet.from_cuts(
            dummy_cut(idx, supervisions=[dummy_supervision(idx)]) for idx in range(begin_id, end_id)
        )


def dummy_recording(unique_id: int) -> Recording:
    return Recording(
        id=f'dummy-recording-{unique_id:04d}',
        sources=[],
        sampling_rate=16000,
        num_samples=16000,
        duration=1.0
    )


def dummy_supervision(unique_id: int, start: float = 0.0, duration: float = 1.0) -> SupervisionSegment:
    return SupervisionSegment(
        id=f'dummy-segment-{unique_id:04d}',
        recording_id=f'dummy-recording-{unique_id:04d}',
        start=start,
        duration=duration,
        text='irrelevant'
    )


def dummy_features(unique_id: int) -> Features:
    return Features(
        recording_id=f'dummy-recording-{unique_id:04d}',
        channels=0,
        start=0.0,
        duration=1.0,
        type='fbank',
        num_frames=100,
        num_features=23,
        sampling_rate=16000,
        storage_type='lilcom_files',
        storage_path='test/fixtures/dummy_feats/storage',
        storage_key='dbf9a0ec-f79d-4eb8-ae83-143a6d5de64d.llc'
    )


def dummy_cut(unique_id: int, start: float = 0.0, duration: float = 1.0, supervisions=None):
    return Cut(
        id=f'dummy-cut-{unique_id:04d}',
        start=start,
        duration=duration,
        channel=0,
        features=dummy_features(unique_id),
        supervisions=supervisions if supervisions is not None else [],
    )


def remove_spaces_from_segment_text(segment):
    if segment.text is None:
        return segment
    return fastcopy(segment, text=segment.text.replace(' ', ''))
