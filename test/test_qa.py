import pytest

from lhotse import Features, RecordingSet, SupervisionSet
from lhotse.qa import remove_missing_recordings_and_supervisions, validate_features
from lhotse.testing.dummies import DummyManifest


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


def test_remove_missing_recordings_and_supervisions():
    recordings = DummyManifest(RecordingSet, begin_id=0, end_id=100)
    supervisions = DummyManifest(SupervisionSet, begin_id=50, end_id=150)
    fix_recs, fix_sups = remove_missing_recordings_and_supervisions(recordings, supervisions)
    expected_ids = [f'dummy-recording-{idx:04d}' for idx in range(50, 100)]
    assert [r.id for r in fix_recs] == expected_ids
    assert [s.recording_id for s in fix_sups] == expected_ids
