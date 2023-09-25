from lhotse import CutSet, RecordingSet, SupervisionSegment
from lhotse.workflows.activity_detection import (
    ActivityDetectionProcessor,
    SileroVAD8k,
    SileroVAD16k,
)


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


def test_silero_vad_in_parallel():
    cuts = CutSet.from_file("test/fixtures/ljspeech/cuts.json")
    recordings = RecordingSet.from_recordings([cut.recording for cut in cuts])
    processor = ActivityDetectionProcessor(SileroVAD8k, num_jobs=2, device="cpu")
    supervisions = processor(recordings)
    newcuts = CutSet.from_manifests(
        recordings=recordings,
        supervisions=supervisions,
    )
    assert len(newcuts) > 0
    for cut in newcuts:
        assert len(cut.supervisions) > 0
        for sup in cut.supervisions:
            assert sup.duration > 0
