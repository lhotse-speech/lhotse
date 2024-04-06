from typing import Optional

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from lhotse import CutSet, SupervisionSet
from lhotse.workflows.meeting_simulation import (
    BaseMeetingSimulator,
    ConversationalMeetingSimulator,
    SpeakerIndependentMeetingSimulator,
)

scipy = pytest.importorskip("scipy", reason="These tests require scipy package to run.")


@pytest.fixture(scope="module")
def cuts():
    libri_cuts = CutSet.from_file("test/fixtures/libri/cuts.json")
    ami_cuts = CutSet.from_file("test/fixtures/ami/cuts.json")
    return (libri_cuts + ami_cuts).cut_into_windows(duration=3.0).to_eager()


@pytest.fixture(scope="module")
def sups():
    return SupervisionSet.from_file("test/fixtures/ami/ES2011a_sups.jsonl.gz")


@settings(deadline=None, print_blob=True, max_examples=50)
@given(
    method=st.one_of([st.just(m) for m in ["independent", "conversational"]]),
    fit_to_supervisions=st.booleans(),
    num_meetings=st.integers(min_value=0, max_value=10),
    num_repeats=st.integers(min_value=1, max_value=5),
    num_speakers_per_meeting=st.integers(min_value=2, max_value=4),
    max_duration_per_speaker=st.floats(min_value=10.0, max_value=20.0),
    max_utterances_per_speaker=st.integers(min_value=0, max_value=5),
    reverberate=st.booleans(),
    seed=st.integers(min_value=0, max_value=2**16 - 1),
    num_jobs=st.integers(min_value=1, max_value=4),
)
def test_simulate_meetings(
    cuts: CutSet,
    sups: SupervisionSet,
    method: str,
    fit_to_supervisions: bool,
    num_meetings: Optional[int],
    num_repeats: int,
    num_speakers_per_meeting: int,
    max_duration_per_speaker: float,
    max_utterances_per_speaker: int,
    reverberate: bool,
    seed: int,
    num_jobs: int,
):
    if method == "independent":
        simulator = SpeakerIndependentMeetingSimulator()
    elif method == "conversational":
        simulator = ConversationalMeetingSimulator()
    else:
        raise ValueError(f"Unknown method: {method}")

    if fit_to_supervisions:
        simulator.fit(sups)

    mixed_cuts = simulator.simulate(
        cuts,
        num_meetings=num_meetings if num_meetings > 0 else None,
        num_repeats=num_repeats,
        num_speakers_per_meeting=num_speakers_per_meeting,
        max_duration_per_speaker=max_duration_per_speaker,
        max_utterances_per_speaker=max_utterances_per_speaker
        if max_utterances_per_speaker > 0
        else None,
        seed=seed,
        num_jobs=num_jobs,
    )

    if reverberate:
        mixed_cuts = simulator.reverberate(mixed_cuts)

    if len(cuts.speakers) >= num_speakers_per_meeting:
        assert len(mixed_cuts) > 0
        mixed_cut = mixed_cuts[0]
        assert mixed_cut.load_audio().shape[1] == mixed_cut.num_samples
    else:
        assert len(mixed_cuts) == 0


def test_base_meeting_simulator_raises():
    with pytest.raises(TypeError):
        BaseMeetingSimulator()
