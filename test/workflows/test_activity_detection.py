import json
import tempfile
from contextlib import suppress
from pathlib import Path

import pytest
import torch
from click.testing import CliRunner

from lhotse import CutSet, RecordingSet, SupervisionSegment, SupervisionSet
from lhotse.bin.modes.workflows import detect_activity as detect_activity_cli
from lhotse.bin.modes.workflows import trim_inactivity as trim_inactivity_cli
from lhotse.workflows.activity_detection import (
    ActivityDetector,
    SileroVAD16k,
    detect_acitvity_segments,
    detect_activity,
    trim_inactivity,
)


def _check_torch_version(greater_than: str):
    with suppress(Exception):
        from pkg_resources import parse_version  # pylint: disable=C0415

        if parse_version(torch.__version__) >= parse_version(greater_than):
            return True
    return False


def test_silero_vad_init():
    if not _check_torch_version("1.12"):
        pytest.skip("torch >= 1.12 is required for this test")

    vad = SileroVAD16k(device="cpu")
    cuts = CutSet.from_file("test/fixtures/ljspeech/cuts.json")
    recording = cuts[0].recording

    activity = detect_acitvity_segments(recording, model=vad)

    assert activity != []
    assert isinstance(activity[0], SupervisionSegment)
    assert activity[0].start != 0
    assert activity[0].duration > 0.3
    assert activity[0].start + activity[0].duration < recording.duration


def test_detect_activity_with_silero_vad_in_parallel():
    if not _check_torch_version("1.12"):
        pytest.skip("torch >= 1.12 is required for this test")

    cuts = CutSet.from_file("test/fixtures/ljspeech/cuts.json")
    recordings = RecordingSet.from_recordings([cut.recording for cut in cuts])

    supervisions = detect_activity(
        recordings,
        detector="silero_vad_8k",
        num_jobs=2,
        device="cpu",
    )

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


def test_detect_activity_workflow_with_silero_vad(temporary_directory: str):
    if not _check_torch_version("1.12"):
        pytest.skip("torch >= 1.12 is required for this test")

    output_manifest_path = Path(temporary_directory) / "temp_output.json"

    runner = CliRunner()

    result = runner.invoke(
        detect_activity_cli,
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


def test_trim_inactivity_with_silero_vad_in_parallel():
    if not _check_torch_version("1.12"):
        pytest.skip("torch >= 1.12 is required for this test")

    cuts = CutSet.from_file("test/fixtures/ljspeech/cuts.json")

    trim, _ = trim_inactivity(
        cuts,
        detector="silero_vad_8k",
        num_jobs=2,
        device="cpu",
    )
    assert trim[0].duration == cuts[0].duration
    assert trim[1].duration < cuts[1].duration


def test_trim_inactivity_workflow_with_silero_vad(temporary_directory: str):
    if not _check_torch_version("1.12"):
        pytest.skip("torch >= 1.12 is required for this test")

    runner = CliRunner()
    libri_cuts_path = "test/fixtures/libri/cuts.json"
    result = runner.invoke(
        trim_inactivity_cli,
        ["--cuts-manifest", libri_cuts_path]
        + ["--output-dir", temporary_directory]
        + ["--model-name", "silero_vad_16k"]
        + ["--device", "cpu"]
        + ["--jobs", "1"]
        + ["--protect-outside"],
    )
    assert result.exit_code == 0

    temp = Path(temporary_directory)
    paths = [str(path.relative_to(temp)) for path in temp.rglob("*")]
    assert "speach-only-report.csv" in paths
    assert "cuts.json.gz" in paths
    assert "recordings.jsonl.gz" in paths
    assert "supervisions.jsonl.gz" in paths

    original = CutSet.from_file(libri_cuts_path)
    trimmed = CutSet.from_file(temp / "cuts.json.gz")
    assert len(original) == len(trimmed)

    assert f"storage/{original[0].id}.flac" in paths

    assert original[0].duration > trimmed[0].duration
    assert original[0].supervisions[0].duration > trimmed[0].supervisions[0].duration
    assert original[0].supervisions[0].text == trimmed[0].supervisions[0].text
    assert trimmed[0].supervisions[0].start == 0.0

    recordings = RecordingSet.from_file(temp / "recordings.jsonl.gz")
    assert len(recordings) == 1
    assert recordings[0].duration == trimmed[0].duration
    assert recordings[0].num_samples == trimmed[0].num_samples

    supervisions = SupervisionSet.from_file(temp / "supervisions.jsonl.gz")
    assert len(supervisions) == 1
    assert supervisions[0].duration == trimmed[0].duration
    assert supervisions[0].start == 0.0
    assert supervisions[0].text == trimmed[0].supervisions[0].text


def test_activity_detector_list_known():
    detectors = ActivityDetector.list_detectors()
    assert "silero_vad_8k" in detectors
    assert "silero_vad_16k" in detectors


def test_activity_detector_not_known():
    with pytest.raises(ValueError):
        ActivityDetector.get_detector("123")
