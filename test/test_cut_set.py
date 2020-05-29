from cytoolz import sliding_window
from pytest import mark

from lhotse.cut import CutSet, Cut
from lhotse.features import Features
from lhotse.supervision import SupervisionSegment


@mark.parametrize(
    ['offset', 'until', 'keep_excessive_supervisions', 'expected_supervision_ids'],
    [
        (0.0, None, True, ['s1', 's2', 's3', 's4']),
        (0.0, None, False, ['s1', 's2', 's3', 's4']),
        (0.0, 0.5, True, ['s1', 's2', 's3', 's4']),
        (0.0, 0.5, False, ['s1', 's2', 's3', 's4']),
        (0.1, None, True, ['s1', 's2', 's3', 's4']),
        (0.1, None, False, ['s2', 's3', 's4']),
        (0.1, 0.4, True, ['s1', 's2', 's3', 's4']),
        (0.1, 0.4, False, ['s2', 's3']),
        (0.2, None, True, ['s2', 's3', 's4']),
        (0.2, None, False, ['s3', 's4']),
        (0.2, 0.4, True, ['s2', 's3', 's4']),
        (0.2, 0.4, False, ['s3']),
        (0.0, 0.1, True, ['s1']),
        (0.0, 0.1, False, []),
        (0.1, 0.2, False, []),
        (0.2, 0.3, False, []),
        (0.3, 0.4, False, []),
        (0.4, 0.5, False, []),
        (0.27, 0.31, False, []),
    ]
)
def test_truncate_cut(offset, until, keep_excessive_supervisions, expected_supervision_ids):
    # TODO: refactor (simplify data) once I'm ready to make a test fixture builder
    cut = Cut(
        id='cut-1',
        channel=0,
        start=0.0,
        duration=0.5,
        features=Features(
            recording_id='recording-1',
            channel_id=0,
            start=0,
            duration=0.5,
            frame_length=25.0,
            frame_shift=10.0,
            storage_type='lilcom',
            storage_path='test/fixtures/dummy_feats/storage/e66b6386-aee5-4a5a-8369-fdde1d2b97c7.llc'
        ),
        supervisions=[
            SupervisionSegment(
                id='s1',
                recording_id='recording-1',
                start=0.0,
                duration=0.2,
                channel_id=0
            ),
            SupervisionSegment(
                id='s2',
                recording_id='recording-1',
                start=0.1,
                duration=0.2,
                channel_id=0
            ),
            SupervisionSegment(
                id='s3',
                recording_id='recording-1',
                start=0.2,
                duration=0.2,
                channel_id=0
            ),
            SupervisionSegment(
                id='s4',
                recording_id='recording-1',
                start=0.3,
                duration=0.2,
                channel_id=0
            )
        ]
    )

    truncated_cut = cut.truncate(
        offset=offset,
        until=until,
        keep_excessive_supervisions=keep_excessive_supervisions
    )
    remaining_supervision_ids = [s.id for s in truncated_cut.supervisions]
    assert remaining_supervision_ids == expected_supervision_ids


def test_cut_set():
    cut_set = CutSet(cuts={})

    # CutSet consists of elements with IDs
    cut = cut_set.cuts['cut-1']

    # Each Cut specifies standard time info
    assert 10.0 == cut.start
    assert 15.0 == cut.duration
    assert 25.0 == cut.end

    # Each Cut consists of supervision segments
    supervisions = cut.supervisions
    assert 3 == len(supervisions)

    # Supervision segments cannot overlap
    for left_segment, right_segment in sliding_window(2, sorted(supervisions, key=lambda s: s.start)):
        assert left_segment.end <= right_segment.start

    # Each Cut contains a feature matrix
    features = cut.features
    # TODO: need to push the "trimming" capability from FeatureSet to Features
    feat_matrix = features.load(
        channel=cut.channel,
        begin=cut.start,
        duration=cut.duration
    )

    # Append Cuts
    another_cut = cut_set.cuts['cut-2']
    concatenated_cuts = cut + another_cut

    # Truncate Cuts
    truncated_cut = cut.truncate(offset=0, until=cut.duration - 0.5)

    # Overlay Cuts - meaning, add their feature matrices and gather supervisions into a common list
    overlayed_cut = cut.overlay(another_cut)
