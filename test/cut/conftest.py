import pytest

from lhotse.audio import AudioSource, Recording
from lhotse.cut import CutSet, MonoCut
from lhotse.features import Features
from lhotse.supervision import SupervisionSegment


@pytest.fixture
def dummy_features():
    return Features(
        recording_id="irrelevant",
        channels=0,
        start=0.0,
        duration=10.0,
        type="fbank",
        num_frames=1000,
        num_features=80,
        sampling_rate=16000,
        storage_type="irrelevant",
        storage_path="irrelevant",
        storage_key="irrelevant",
        frame_shift=0.01,
    )


@pytest.fixture
def dummy_recording():
    return Recording(
        id="irrelevant",
        sources=[AudioSource(type="file", channels=[0], source="irrelevant")],
        sampling_rate=16000,
        num_samples=160000,
        duration=10.0,
    )


@pytest.fixture
def cut1(dummy_features, dummy_recording):
    return MonoCut(
        id="cut-1",
        start=0.0,
        duration=10.0,
        channel=0,
        features=dummy_features,
        recording=dummy_recording,
        supervisions=[
            SupervisionSegment(
                id="sup-1", recording_id="irrelevant", start=0.5, duration=6.0
            ),
            SupervisionSegment(
                id="sup-2", recording_id="irrelevant", start=7.0, duration=2.0
            ),
        ],
    )


@pytest.fixture
def cut2(dummy_features, dummy_recording):
    return MonoCut(
        id="cut-2",
        start=180.0,
        duration=10.0,
        channel=0,
        features=dummy_features,
        recording=dummy_recording,
        supervisions=[
            SupervisionSegment(
                id="sup-3", recording_id="irrelevant", start=3.0, duration=2.5
            )
        ],
    )


@pytest.fixture
def cut_set(cut1, cut2):
    return CutSet.from_cuts([cut1, cut2])
