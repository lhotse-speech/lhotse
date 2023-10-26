import json
import tempfile
from contextlib import suppress
from functools import partial
from itertools import chain
from pathlib import Path

import pytest
import torch
from click.testing import CliRunner

from lhotse import CutSet, RecordingSet, SupervisionSegment, SupervisionSet
from lhotse.bin.modes.workflows import activity_detection
from lhotse.parallel import ParallelExecutor
from lhotse.workflows.activity_detection import SileroVAD8k, SileroVAD16k


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
    activity = vad(recording)
    assert activity != []
    assert isinstance(activity[0], SupervisionSegment)
    assert activity[0].start != 0
    assert activity[0].duration > 0.3
    assert activity[0].start + activity[0].duration < recording.duration


def test_silero_vad_in_parallel():
    if not _check_torch_version("1.12"):
        pytest.skip("torch >= 1.12 is required for this test")

    cuts = CutSet.from_file("test/fixtures/ljspeech/cuts.json")
    recordings = RecordingSet.from_recordings([cut.recording for cut in cuts])
    processor = ParallelExecutor(partial(SileroVAD8k, device="cpu"), num_jobs=2)
    supervisions = SupervisionSet.from_segments(
        chain.from_iterable(processor(recordings))
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


def test_silero_vad_workflow_simple(temporary_directory: str):
    if not _check_torch_version("1.12"):
        pytest.skip("torch >= 1.12 is required for this test")

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
