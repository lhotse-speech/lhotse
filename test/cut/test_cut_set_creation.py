import pytest

from lhotse.cut import make_windowed_cuts_from_features
from lhotse.features import FeatureSet, Features


def features(rec_id, start, duration):
    """Helper method for fixture readability (specify only relevant attributes)."""
    return Features(recording_id=rec_id, channels=0, start=start, duration=duration, sampling_rate=16000,
                    type='irrelevant', num_frames=round(duration / 0.01), num_features=23,
                    storage_type='irrelevant', storage_path='irrelevant', storage_key='irrelevant')


@pytest.fixture
def feature_set():
    return FeatureSet(
        features=[
            features('rec-1', 0.0, 600.0),
            features('rec-2', 0.0, 357.0)
        ]
    )


# noinspection PyMethodMayBeStatic
class TestMakeWindowedCutsFromFeatures:
    def test_full_shift_no_shorter_cuts(self, feature_set):
        cut_set = make_windowed_cuts_from_features(
            feature_set=feature_set,
            cut_duration=5.0
        )

        assert len(cut_set) == 191
        assert len([c for c in cut_set if c.recording_id == 'rec-1']) == 120
        assert len([c for c in cut_set if c.recording_id == 'rec-2']) == 71

        assert all(c.duration == 5.0 for c in cut_set)

    def test_full_shift_with_shorter_cuts(self, feature_set):
        cut_set = make_windowed_cuts_from_features(
            feature_set=feature_set,
            cut_duration=5.0,
            keep_shorter_windows=True
        )

        assert len(cut_set) == 192
        assert len([c for c in cut_set if c.recording_id == 'rec-1']) == 120
        assert len([c for c in cut_set if c.recording_id == 'rec-2']) == 72

        assert not all(c.duration == 5.0 for c in cut_set)

    def test_half_shift_no_shorter_cuts(self, feature_set):
        cut_set = make_windowed_cuts_from_features(
            feature_set=feature_set,
            cut_duration=5.0,
            cut_shift=2.5
        )

        assert len(cut_set) == 380
        # below, the last window is only 2.5s duration
        assert len([c for c in cut_set if c.recording_id == 'rec-1']) == 239
        # below, the last two windows are 2.0s and 4.5s duration
        assert len([c for c in cut_set if c.recording_id == 'rec-2']) == 141

        assert all(c.duration == 5.0 for c in cut_set)

    def test_half_shift_with_shorter_cuts(self, feature_set):
        cut_set = make_windowed_cuts_from_features(
            feature_set=feature_set,
            cut_duration=5.0,
            cut_shift=2.5,
            keep_shorter_windows=True
        )

        assert len(cut_set) == 383
        assert len([c for c in cut_set if c.recording_id == 'rec-1']) == 240
        assert len([c for c in cut_set if c.recording_id == 'rec-2']) == 143

        assert not all(c.duration == 5.0 for c in cut_set)
