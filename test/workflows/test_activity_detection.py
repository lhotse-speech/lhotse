from lhotse import CutSet, SupervisionSegment
from lhotse.workflows.activity_detection import SileroVAD16k


def test_silero_vad_init():
    vad = SileroVAD16k(device="cpu")
    cuts = CutSet.from_file("test/fixtures/ljspeech/cuts.json")
    recording = cuts[0].recording
    activity = vad(recording)
    assert activity != []
    assert isinstance(activity[0], SupervisionSegment)
    assert activity[0].start != 0
    assert activity[0].duration > 0.3
    assert activity[0].start + activity[0].duration < recording.duration
