"""Tests for loading manifests from stdin (``"-"``).

See https://github.com/lhotse-speech/lhotse/issues/810.
"""

import io
import json
import sys

import pytest

from lhotse import RecordingSet
from lhotse.serialization import load_manifest_lazy_or_eager
from lhotse.testing.dummies import dummy_recording, dummy_supervision


def _redirect_stdin(monkeypatch, payload: str):
    monkeypatch.setattr(sys, "stdin", io.StringIO(payload))


def _to_jsonl(items) -> str:
    return "\n".join(json.dumps(item.to_dict()) for item in items) + "\n"


def test_load_manifest_from_stdin_returns_full_manifest(monkeypatch):
    recordings = [dummy_recording(i) for i in range(8)]
    _redirect_stdin(monkeypatch, _to_jsonl(recordings))

    manifest = load_manifest_lazy_or_eager("-")

    assert isinstance(manifest, RecordingSet)
    assert len(manifest) == 8
    assert [r.id for r in manifest] == [r.id for r in recordings]


def test_load_manifest_from_stdin_supports_split(monkeypatch):
    """Regression test for #810: ``load + split`` used to fail because the lazy
    loader tried to re-read stdin to materialize the iterator.
    """
    recordings = [dummy_recording(i) for i in range(8)]
    _redirect_stdin(monkeypatch, _to_jsonl(recordings))

    manifest = load_manifest_lazy_or_eager("-")
    parts = manifest.split(num_splits=4)

    assert len(parts) == 4
    assert sum(len(p) for p in parts) == len(recordings)


def test_load_manifest_from_stdin_can_be_iterated_twice(monkeypatch):
    recordings = [dummy_recording(i) for i in range(3)]
    _redirect_stdin(monkeypatch, _to_jsonl(recordings))

    manifest = load_manifest_lazy_or_eager("-")
    first_pass = [r.id for r in manifest]
    second_pass = [r.id for r in manifest]

    assert first_pass == second_pass == [r.id for r in recordings]


def test_load_manifest_from_stdin_detects_supervision_set(monkeypatch):
    supervisions = [dummy_supervision(i) for i in range(2)]
    _redirect_stdin(monkeypatch, _to_jsonl(supervisions))

    manifest = load_manifest_lazy_or_eager("-")

    from lhotse import SupervisionSet

    assert isinstance(manifest, SupervisionSet)
    assert len(manifest) == 2


def test_load_manifest_from_stdin_returns_none_for_empty_input(monkeypatch):
    _redirect_stdin(monkeypatch, "")
    assert load_manifest_lazy_or_eager("-") is None


def test_load_manifest_from_stdin_with_explicit_manifest_cls(monkeypatch):
    recordings = [dummy_recording(i) for i in range(2)]
    _redirect_stdin(monkeypatch, _to_jsonl(recordings))

    manifest = load_manifest_lazy_or_eager("-", manifest_cls=RecordingSet)

    assert isinstance(manifest, RecordingSet)
    assert len(manifest) == 2


def test_load_manifest_from_stdin_garbage_raises(monkeypatch):
    _redirect_stdin(monkeypatch, "this is not json\n")
    with pytest.raises(Exception):
        load_manifest_lazy_or_eager("-")
