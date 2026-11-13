import os
import ssl
import tempfile
from pathlib import Path

import pytest

from lhotse.audio import RecordingSet
from lhotse.cut import CutSet, MultiCut
from lhotse.recipes.utils import (
    create_cuts_per_recording,
    manifests_exist,
    read_manifests_if_cached,
)
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


def test_create_cuts_per_recording_spans_recordings_with_all_supervisions(
    recording_set: RecordingSet, supervision_set: SupervisionSet
):
    cuts = create_cuts_per_recording(recording_set, supervision_set)

    assert not cuts.is_lazy
    assert list(cuts.ids) == list(recording_set.ids)
    for cut, recording in zip(cuts, recording_set):
        assert isinstance(cut, MultiCut)
        assert cut.start == 0
        assert cut.duration == recording.duration
        assert cut.channel == recording.channel_ids
        assert cut.recording == recording
        assert cut.supervisions == [
            s for s in supervision_set if s.recording_id == recording.id
        ]
    assert [len(c.supervisions) for c in cuts] == [2, 2, 0, 0]


def test_create_cuts_per_recording_without_supervisions(recording_set: RecordingSet):
    cuts = create_cuts_per_recording(recording_set)

    assert list(cuts.ids) == list(recording_set.ids)
    assert all(c.supervisions == [] for c in cuts)


def test_create_cuts_per_recording_writes_lazy_cuts(
    tmp_path: Path, recording_set: RecordingSet, supervision_set: SupervisionSet
):
    recordings_path = tmp_path / "recordings_dev.jsonl.gz"
    recording_set.to_file(recordings_path)
    cuts_path = tmp_path / "cuts_dev.jsonl.gz"

    cuts = create_cuts_per_recording(
        RecordingSet.from_jsonl_lazy(recordings_path),
        supervision_set,
        output_path=cuts_path,
    )

    assert cuts.is_lazy
    assert cuts_path.is_file()
    assert list(cuts) == list(create_cuts_per_recording(recording_set, supervision_set))


def test_read_manifests_if_cached_requires_and_lazily_reads_cuts(
    tmp_path: Path, recording_set: RecordingSet, supervision_set: SupervisionSet
):
    recording_set.to_file(tmp_path / "corpus_recordings_dev.jsonl.gz")
    supervision_set.to_file(tmp_path / "corpus_supervisions_dev.jsonl.gz")
    assert not manifests_exist("dev", output_dir=tmp_path, prefix="corpus")

    cuts = create_cuts_per_recording(
        recording_set,
        supervision_set,
        output_path=tmp_path / "corpus_cuts_dev.jsonl.gz",
    )
    assert manifests_exist("dev", output_dir=tmp_path, prefix="corpus")

    cached = read_manifests_if_cached(["dev"], output_dir=tmp_path, prefix="corpus")
    assert sorted(cached["dev"]) == ["cuts", "recordings", "supervisions"]
    assert not cached["dev"]["recordings"].is_lazy
    assert isinstance(cached["dev"]["cuts"], CutSet)
    assert cached["dev"]["cuts"].is_lazy
    assert list(cached["dev"]["cuts"]) == list(cuts)


def test_ssl_unverified_context():
    assert ssl._create_default_https_context is ssl._create_unverified_context