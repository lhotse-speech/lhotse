import pytest

from lhotse.cut import Cut, CutSet
from lhotse.features import Features
from lhotse.supervision import SupervisionSegment


@pytest.fixture
def dummy_features():
    return Features(
        recording_id='irrelevant', channel_id=0, start=0.0, duration=10.0,
        type='fbank', num_frames=1000, num_features=80,
        storage_type='irrelevant', storage_path='irrelevant'
    )


@pytest.fixture
def cut1(dummy_features):
    return Cut(id='cut-1', start=0.0, duration=10.0, features=dummy_features, supervisions=[
        SupervisionSegment(id='sup-1', recording_id='irrelevant', start=0.5, duration=6.0),
        SupervisionSegment(id='sup-2', recording_id='irrelevant', start=7.0, duration=2.0)
    ])


@pytest.fixture
def cut2(dummy_features):
    return Cut(id='cut-2', start=180.0, duration=10.0, features=dummy_features, supervisions=[
        SupervisionSegment(id='sup-3', recording_id='irrelevant', start=3.0, duration=2.5)
    ])


@pytest.fixture
def cut_set(cut1, cut2):
    return CutSet({cut.id: cut for cut in [cut1, cut2]})
