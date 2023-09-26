import json
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from lhotse import CutSet, RecordingSet, SupervisionSegment
from lhotse.bin.modes.workflows import activity_detection
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


@pytest.fixture
def temporary_directory():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


def test_silero_vad_workflow_simple(temporary_directory: str):
    output_manifest_path = Path(temporary_directory) / "temp_output.json"

    runner = CliRunner()

    result = runner.invoke(
        activity_detection,
        ["--recordings-manifest", "test/fixtures/libri/audio.json"]
        + ["--output-supervisions-manifest", str(output_manifest_path)]
        + ["--model-name", "silero_vad_16k"]
        + ["--device", "cpu"]
        + ["--jobs", "1"]
        # "--force_download",
    )

    assert result.exit_code == 0
    assert "Results saved to" in result.output
    with open(output_manifest_path, encoding="utf-8") as file:
        supervisions = json.load(file)
    for i, segment in enumerate(supervisions):
        assert segment["channel"] == 0
        assert segment["start"] >= 0.1
        assert segment["duration"] > 0
        assert segment["recording_id"] == "recording-1"
        assert segment["id"] == f"recording-1-silero_vad_16k-0-{i:05}"
