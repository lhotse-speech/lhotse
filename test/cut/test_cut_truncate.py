from math import isclose

import pytest

from lhotse.cut import Cut, MixedCut, MixTrack
from lhotse.features import Features
from lhotse.supervision import SupervisionSegment
from lhotse.test_utils import dummy_cut


@pytest.fixture
def overlapping_supervisions_cut():
    return Cut(
        id='cut-1',
        start=0.0,
        duration=0.5,
        features=Features(
            recording_id='recording-1',
            channel_id=0,
            start=0,
            duration=0.5,
            type='fbank',
            num_frames=50,
            num_features=80,
            storage_type='lilcom',
            storage_path='test/fixtures/dummy_feats/storage/e66b6386-aee5-4a5a-8369-fdde1d2b97c7.llc'
        ),
        supervisions=[
            SupervisionSegment(id='s1', recording_id='recording-1', start=0.0, duration=0.2),
            SupervisionSegment(id='s2', recording_id='recording-1', start=0.1, duration=0.2),
            SupervisionSegment(id='s3', recording_id='recording-1', start=0.2, duration=0.2),
            SupervisionSegment(id='s4', recording_id='recording-1', start=0.3, duration=0.2)
        ]
    )


@pytest.mark.parametrize(
    ['offset', 'until', 'keep_excessive_supervisions', 'expected_duration', 'expected_supervision_ids'],
    [
        (0.0, None, True, 0.5, ['s1', 's2', 's3', 's4']),
        (0.0, None, False, 0.5, ['s1', 's2', 's3', 's4']),
        (0.0, 0.5, True, 0.5, ['s1', 's2', 's3', 's4']),
        (0.0, 0.5, False, 0.5, ['s1', 's2', 's3', 's4']),
        (0.1, None, True, 0.4, ['s1', 's2', 's3', 's4']),
        (0.1, None, False, 0.4, ['s2', 's3', 's4']),
        (0.0, 0.4, True, 0.4, ['s1', 's2', 's3', 's4']),
        (0.0, 0.4, False, 0.4, ['s1', 's2', 's3']),
        (0.1, 0.4, True, 0.3, ['s1', 's2', 's3', 's4']),
        (0.1, 0.4, False, 0.3, ['s2', 's3']),
        (0.1, 0.2, True, 0.1, ['s1', 's2']),
        (0.1, 0.2, False, 0.1, []),
        (0.2, None, True, 0.3, ['s2', 's3', 's4']),
        (0.2, None, False, 0.3, ['s3', 's4']),
        (0.2, 0.4, True, 0.2, ['s2', 's3', 's4']),
        (0.2, 0.4, False, 0.2, ['s3']),
        (0.0, 0.1, True, 0.1, ['s1']),
        (0.0, 0.1, False, 0.1, []),
        (0.1, 0.2, False, 0.1, []),
        (0.2, 0.3, False, 0.1, []),
        (0.3, 0.4, False, 0.1, []),
        (0.4, 0.5, False, 0.1, []),
        (0.27, 0.31, False, 0.04, []),
    ]
)
def test_truncate_cut(
        offset,
        until,
        keep_excessive_supervisions,
        expected_duration,
        expected_supervision_ids,
        overlapping_supervisions_cut
):
    truncated_cut = overlapping_supervisions_cut.truncate(
        offset=offset,
        until=until,
        keep_excessive_supervisions=keep_excessive_supervisions
    )
    remaining_supervision_ids = [s.id for s in truncated_cut.supervisions]
    assert remaining_supervision_ids == expected_supervision_ids
    assert isclose(truncated_cut.duration, expected_duration)


@pytest.fixture
def simple_mixed_cut():
    return MixedCut(
        id='simple-mixed-cut',
        tracks=[
            MixTrack(cut=dummy_cut('cut0', duration=10.0)),
            MixTrack(cut=dummy_cut('cut1', duration=10.0), offset=5.0),
        ]
    )


def test_truncate_mixed_cut_without_args(simple_mixed_cut):
    truncated_cut = simple_mixed_cut.truncate()
    assert truncated_cut.duration == 15.0


def test_truncate_mixed_cut_with_small_offset(simple_mixed_cut):
    truncated_cut = simple_mixed_cut.truncate(offset=1.0)

    assert len(truncated_cut.tracks) == 2

    assert truncated_cut.tracks[0].offset == 0.0
    assert truncated_cut.tracks[0].cut.start == 1.0
    assert truncated_cut.tracks[0].cut.duration == 9.0
    assert truncated_cut.tracks[0].cut.end == 10.0

    assert truncated_cut.tracks[1].offset == 4.0
    assert truncated_cut.tracks[1].cut.start == 0.0
    assert truncated_cut.tracks[1].cut.duration == 10.0
    assert truncated_cut.tracks[1].cut.end == 10.0

    assert truncated_cut.duration == 14.0


def test_truncate_mixed_cut_with_offset_exceeding_first_track(simple_mixed_cut):
    truncated_cut = simple_mixed_cut.truncate(offset=11.0)

    assert len(truncated_cut.tracks) == 1

    assert truncated_cut.tracks[0].cut.id == 'cut2'
    assert truncated_cut.tracks[0].offset == 0.0
    assert truncated_cut.tracks[0].cut.start == 6.0
    assert truncated_cut.tracks[0].cut.duration == 4.0
    assert truncated_cut.tracks[0].cut.end == 10.0

    assert truncated_cut.duration == 4.0


def test_truncate_mixed_cut_decreased_duration(simple_mixed_cut):
    truncated_cut = simple_mixed_cut.truncate(duration=14.0)

    assert len(truncated_cut.tracks) == 2

    assert truncated_cut.tracks[0].offset == 0.0
    assert truncated_cut.tracks[0].cut.start == 0.0
    assert truncated_cut.tracks[0].cut.duration == 10.0
    assert truncated_cut.tracks[0].cut.end == 10.0

    assert truncated_cut.tracks[1].offset == 5.0
    assert truncated_cut.tracks[1].cut.start == 0.0
    assert truncated_cut.tracks[1].cut.duration == 9.0
    assert truncated_cut.tracks[1].cut.end == 9.0

    assert truncated_cut.duration == 14.0


def test_truncate_mixed_cut_decreased_duration_removing_last_cut(simple_mixed_cut):
    truncated_cut = simple_mixed_cut.truncate(duration=4.0)

    assert len(truncated_cut.tracks) == 1

    assert truncated_cut.tracks[0].cut.id == 'cut1'
    assert truncated_cut.tracks[0].offset == 0.0
    assert truncated_cut.tracks[0].cut.start == 0.0
    assert truncated_cut.tracks[0].cut.duration == 4.0
    assert truncated_cut.tracks[0].cut.end == 4.0

    assert truncated_cut.duration == 4.0


def test_truncate_mixed_cut_with_small_offset_and_duration(simple_mixed_cut):
    truncated_cut = simple_mixed_cut.truncate(offset=1.0, duration=13.0)

    assert len(truncated_cut.tracks) == 2

    assert truncated_cut.tracks[0].offset == 0.0
    assert truncated_cut.tracks[0].cut.start == 1.0
    assert truncated_cut.tracks[0].cut.duration == 9.0
    assert truncated_cut.tracks[0].cut.end == 10.0

    assert truncated_cut.tracks[1].offset == 4.0
    assert truncated_cut.tracks[1].cut.start == 0.0
    assert truncated_cut.tracks[1].cut.duration == 9.0
    assert truncated_cut.tracks[1].cut.end == 9.0

    assert truncated_cut.duration == 13.0


def test_truncate_cut_set_offset_start(cut_set):
    truncated_cut_set = cut_set.truncate(max_duration=5, offset_type='start')
    for cut in truncated_cut_set:
        assert isclose(cut.duration, 5.0)
        assert isclose(cut.start, 0.0)


def test_truncate_cut_set_offset_end(cut_set):
    truncated_cut_set = cut_set.truncate(max_duration=5, offset_type='end')
    for cut in truncated_cut_set:
        assert isclose(cut.duration, 5.0)
        assert isclose(cut.start, 5.0)


def test_truncate_cut_set_offset_random(cut_set):
    truncated_cut_set = cut_set.truncate(max_duration=5, offset_type='random')
    for cut in truncated_cut_set:
        assert isclose(cut.duration, 5.0)
        assert 0.0 <= cut.start <= 5.0
    # Check that "cut.start" is not the same in every cut
    assert len(set(cut.start for cut in truncated_cut_set)) > 1


def test_truncate_cut_set_offset_start(cut_set):
    truncated_cut_set = cut_set.truncate(max_duration=5, offset_type='start')
    for cut in truncated_cut_set:
        assert isclose(cut.duration, 5.0)
        assert isclose(cut.start, 0.0)
