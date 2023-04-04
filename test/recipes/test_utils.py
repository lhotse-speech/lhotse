import os
import tempfile
from pathlib import Path

import pytest

from lhotse.audio import RecordingSet
from lhotse.recipes.utils import read_manifests_if_cached
from lhotse.supervision import SupervisionSet


@pytest.fixture
def recording_set() -> RecordingSet:
    return RecordingSet.from_json("test/fixtures/audio.json")


@pytest.fixture
def supervision_set() -> SupervisionSet:
    return SupervisionSet.from_json(
        "test/fixtures/supervision.json"
    ).with_alignment_from_ctm("test/fixtures/supervision.ctm")


def test_read_manifests_if_cached(
    recording_set: RecordingSet, supervision_set: SupervisionSet
):
    tmp_test_dir = Path(f"{tempfile.gettempdir()}/lhotse_test_read_manifests_if_cached")
    if not tmp_test_dir.exists():
        tmp_test_dir.mkdir()
    data_part = "dev"
    suffix = "jsonl.gz"
    tmp_recording_set_file = tmp_test_dir / f"recordings_{data_part}.{suffix}"
    tmp_supervision_set_file = tmp_test_dir / f"supervisions_{data_part}.{suffix}"
    recording_set.to_jsonl(tmp_recording_set_file)
    supervision_set.to_jsonl(tmp_supervision_set_file)

    try:
        cached_manifests = read_manifests_if_cached(
            [data_part], output_dir=tmp_test_dir
        )
        assert data_part in cached_manifests
        assert cached_manifests[data_part]["recordings"] == recording_set

        cached_manifests = read_manifests_if_cached(data_part, output_dir=tmp_test_dir)
        assert data_part in cached_manifests
        assert cached_manifests[data_part]["recordings"] == recording_set
    finally:
        os.remove(tmp_recording_set_file)
        os.remove(tmp_supervision_set_file)
