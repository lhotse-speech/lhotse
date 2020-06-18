from typing import Type, Tuple

from lhotse.audio import AudioSet, Recording
from lhotse.features import FeatureSet, Features
from lhotse.manipulation import Manifest
from lhotse.supervision import SupervisionSet, SupervisionSegment


# noinspection PyPep8Naming
def DummyManifest(type_: Type, *, begin_id: int, end_id: int) -> Manifest:
    if type_ == AudioSet:
        return AudioSet(recordings=dict([dummy_recording(idx) for idx in range(begin_id, end_id)]))
    if type_ == SupervisionSet:
        return SupervisionSet(segments=dict([dummy_supervision(idx) for idx in range(begin_id, end_id)]))
    if type_ == FeatureSet:
        # noinspection PyTypeChecker
        return FeatureSet(
            features=[dummy_features(idx) for idx in range(begin_id, end_id)],
            feature_extractor='irrelevant'
        )


def dummy_recording(unique_id: int) -> Tuple[str, Recording]:
    rec_id = f'dummy-recording-{unique_id:04d}'
    return rec_id, Recording(
        id=rec_id,
        sources=[],
        sampling_rate=16000,
        num_samples=16000,
        duration_seconds=1.0
    )


def dummy_supervision(unique_id: int) -> Tuple[str, SupervisionSegment]:
    seg_id = f'dummy-segment-{unique_id:04d}'
    return seg_id, SupervisionSegment(
        id=seg_id,
        recording_id=f'dummy-recording',
        start=0.0,
        duration=1.0
    )


def dummy_features(unique_id: int) -> Features:
    return Features(
        recording_id=f'dummy-recording-{unique_id:04d}',
        channel_id=0,
        start=0.0,
        duration=1.0,
        type='fbank',
        num_frames=100,
        num_features=20,
        storage_type='lilcom',
        storage_path='irrelevant'
    )
