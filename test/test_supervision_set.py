from functools import lru_cache

from lhotse.supervision import SupervisionSet, SupervisionSegment


@lru_cache(1)
def load_supervision_set():
    return SupervisionSet.from_yaml('test/fixtures/supervision.yaml')


def test_supervision_segment_with_full_metadata():
    supervision_set = load_supervision_set()
    segment = supervision_set.get('segment-1')
    assert 'segment-1' == segment.id
    assert 'recording-1' == segment.recording_id
    assert 0.1 == segment.start
    assert 0.3 == segment.duration
    assert 'transcript of the first segment' == segment.text
    assert 'english' == segment.language
    assert 'Norman Dyhrentfurth' == segment.speaker


def test_supervision_segment_with_no_metadata():
    supervision_set = load_supervision_set()
    segment = supervision_set.get('segment-2')
    assert 'segment-2' == segment.id
    assert 'recording-1' == segment.recording_id
    assert 0.5 == segment.start
    assert 0.4 == segment.duration
    assert segment.text is None
    assert segment.language is None
    assert segment.speaker is None


def test_create_supervision_segment_with_minimum_metadata():
    SupervisionSegment(id='X', recording_id='X', start=0.0, duration=0.1)


def test_create_supervision_segment_with_all_metadata():
    SupervisionSegment(
        id='X',
        recording_id='X',
        start=0.0,
        duration=0.1,
        text='wysokie szczyty',
        language='polish',
        speaker='Janusz'
    )


def test_supervision_set_iteration():
    supervision_set = SupervisionSet(
        segments={
            'X': SupervisionSegment(id='X', recording_id='X', start=2.0, duration=2.5),
            'Y': SupervisionSegment(id='Y', recording_id='X', start=5.0, duration=5.0),
        }
    )
    assert 2 == len(supervision_set)
    assert 2 == len(list(supervision_set))


def test_supervision_set_serialization():
    supervision_set = SupervisionSet(
        segments={
            'segment-1': SupervisionSegment(
                id='segment-1',
                recording_id='recording-1',
                start=0.1,
                duration=0.3,
                text='transcript of the first segment',
                language='english',
                speaker='Norman Dyhrentfurth'
            )
        }
    )
    supervision_set.to_yaml('.test.supervision.yaml')
    restored = supervision_set.from_yaml('.test.supervision.yaml')
    assert supervision_set == restored
