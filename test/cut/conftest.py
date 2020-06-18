import pytest

from lhotse.cut import Cut, CutSet, MixedCut, MixTrack
from lhotse.features import Features
from lhotse.supervision import SupervisionSegment


@pytest.fixture
def dummy_features():
    return Features(
        recording_id='irrelevant', channel_id=0, start=0.0, duration=10.0,
        frame_length='irrelevant', frame_shift='irrelevant', storage_type='irrelevant', storage_path='irrelevant'
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
    return CutSet.from_cuts([cut1, cut2])


@pytest.fixture()
def mixed_cut(cut1, cut2):
    return MixedCut(
        id='mixed-cut-id',
        tracks=[
            MixTrack(cut_id=cut1.id),
            MixTrack(cut_id=cut2.id, offset=1.0, snr=10),
        ],
        start=0.0,
        duration=11.0
    )


@pytest.fixture
def cut_set_with_mixed_cut(cut1, cut2, mixed_cut):
    return CutSet.from_cuts([cut1, cut2, mixed_cut])
