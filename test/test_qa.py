import pytest

from lhotse import Features
from lhotse.qa import validate_features


def test_validate_features_consistent_num_frames_does_not_raise():
    manifest = Features(
        type='irrelevant',
        num_frames=100,
        num_features=40,
        frame_shift=0.01,
        sampling_rate=16000,
        start=0.0,
        duration=1.0,
        storage_type='irrelevant',
        storage_path='irrelevant',
        storage_key='irrelevant',
    )
    validate_features(manifest)


def test_validate_features_inconsistent_num_frames_raises():
    manifest = Features(
        type='irrelevant',
        num_frames=101,
        num_features=40,
        frame_shift=0.01,
        sampling_rate=16000,
        start=0.0,
        duration=1.0,
        storage_type='irrelevant',
        storage_path='irrelevant',
        storage_key='irrelevant',
    )
    with pytest.raises(AssertionError):
        validate_features(manifest)
