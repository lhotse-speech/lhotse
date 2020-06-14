from math import isclose
from tempfile import NamedTemporaryFile

import numpy as np
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
        (0.0, 0.4, True, ['s1', 's2', 's3', 's4']),
        (0.0, 0.4, False, ['s1', 's2', 's3']),
        (0.1, 0.4, True, ['s1', 's2', 's3', 's4']),
        (0.1, 0.4, False, ['s2', 's3']),
        (0.1, 0.2, True, ['s1', 's2']),
        (0.1, 0.2, False, []),
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


def test_cut_set_serialization():
    # TODO: include OverlayedCut
    cut_set = CutSet(cuts={
        'cut-1': Cut(
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
                    duration=0.5,
                    channel_id=0
                ),
            ]
        )
    })
    with NamedTemporaryFile() as f:
        cut_set.to_yaml(f.name)
        restored = cut_set.from_yaml(f.name)
    assert cut_set == restored


def test_mixed_cut_load_features():
    cut_set = CutSet.from_yaml('test/fixtures/mix_cut_test/overlayed_cut_manifest.yml')

    mixed_cut = cut_set.cuts['mixed-cut-id']
    assert mixed_cut.offset_right_by == 3.89

    ingredient_cut1 = cut_set.cuts[mixed_cut.left_cut_id]
    assert ingredient_cut1.duration == 7.78

    ingredient_cut2 = cut_set.cuts[mixed_cut.right_cut_id]
    assert ingredient_cut2.duration == 9.705

    feats = mixed_cut.with_cut_set(cut_set).load_features()
    expected_duration = mixed_cut.offset_right_by + ingredient_cut2.duration
    assert isclose(expected_duration, 13.595)
    expected_frame_count = 1358
    assert feats.shape[0] == expected_frame_count

    same_mixed_cut = ingredient_cut1.overlay(
        ingredient_cut2,
        offset_other_by=0.5 * ingredient_cut1.duration,
        snr=20.0
    )
    same_feats = same_mixed_cut.with_cut_set(cut_set).load_features()

    np.testing.assert_almost_equal(feats, same_feats)


def test_append_cut():
    cut_set = CutSet.from_yaml('test/fixtures/mix_cut_test/overlayed_cut_manifest.yml')
    ingredient_cut1 = cut_set.cuts['0c5fdf79-efe7-4d45-b612-3d90d9af8c4e']
    assert ingredient_cut1.duration == 7.78
    ingredient_cut2 = cut_set.cuts['78bef88d-e62e-4cfa-9946-a1311442c6f7']
    assert ingredient_cut2.duration == 9.705
    appended_cut = ingredient_cut1.append(ingredient_cut2, snr=5.0).with_cut_set(cut_set)
    assert appended_cut.duration == 7.78 + 9.705
