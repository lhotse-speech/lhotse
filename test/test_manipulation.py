from typing import Tuple, Type

import pytest
from pytest import mark

from lhotse.audio import AudioSet, Recording
from lhotse.features import Features, FeatureSet
from lhotse.manipulation import split, Manifest
from lhotse.supervision import SupervisionSegment, SupervisionSet


@mark.parametrize('manifest_type', [AudioSet, SupervisionSet, FeatureSet])
def test_split_even(manifest_type):
    manifest = dummy(manifest_type, 0, 100)
    manifest_subsets = split(manifest, num_splits=2)
    assert len(manifest_subsets) == 2
    assert manifest_subsets[0] == dummy(manifest_type, 0, 50)
    assert manifest_subsets[1] == dummy(manifest_type, 50, 100)


@mark.parametrize('manifest_type', [AudioSet, SupervisionSet, FeatureSet])
def test_split_odd(manifest_type):
    manifest = dummy(manifest_type, 0, 100)
    manifest_subsets = split(manifest, num_splits=3)
    assert len(manifest_subsets) == 2
    assert manifest_subsets[0] == dummy(manifest_type, 0, 34)
    assert manifest_subsets[1] == dummy(manifest_type, 34, 68)
    assert manifest_subsets[2] == dummy(manifest_type, 68, 100)


@mark.parametrize('manifest_type', [AudioSet, SupervisionSet, FeatureSet])
def test_cannot_split_to_more_chunks_than_items(manifest_type):
    manifest = dummy(manifest_type, 0, 1)
    with pytest.raises(ValueError):
        split(manifest, num_splits=2)


def dummy(manifest_type: Type, begin_idx: int, end_idx: int) -> Manifest:
    if manifest_type == AudioSet:
        return AudioSet(recordings=dict([dummy_recording(idx) for idx in range(begin_idx, end_idx)]))
    if manifest_type == SupervisionSet:
        return SupervisionSet(segments=dict([dummy_supervision(idx) for idx in range(begin_idx, end_idx)]))
    if manifest_type == FeatureSet:
        # noinspection PyTypeChecker
        return FeatureSet(
            features=[dummy_features(idx) for idx in range(begin_idx, end_idx)],
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
        recording_id=f'dummy-recording-{unique_id: 04d}',
        channel_id=0,
        start=0.0,
        duration=1.0,
        storage_type='lilcom',
        storage_path='irrelevant'
    )
