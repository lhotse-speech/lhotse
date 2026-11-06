"""Directory discovery shared by streaming, sidecar and pack readers."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from lhotse.serialization import AIStoreIOBackend
from lhotse.shar.readers.discovery import _discover_fields
from lhotse.shar.readers.lazy import LazySharIterator


@pytest.mark.parametrize(
    "scheme,backend",
    [(scheme, backend) for scheme in ["s3", "s3a", "s3n"] for backend in ["s3", "ais"]]
    + [("ais", "ais")],
)
@pytest.mark.parametrize("prefix", ["", "shar/"])
def test_remote_discovery(monkeypatch, scheme, prefix, backend):
    keys = [
        prefix + name
        for name in [
            "cuts.000001.jsonl.gz",
            "cuts.000000.jsonl.gz",
            "audio.000000.tar",
            "audio.000001.tar",
            "audio_extra.000000.tar",
            "audio_extra.000001.tar",
            "cuts.000000.jsonl.gz.idx",
            "cuts.000000.jsonl.gz.gzidx",
            "audio.000000.tar.idx",
            "dataset.idxpack",
            "nested/ignored.tar",
            "",
        ]
    ]
    monkeypatch.setattr(AIStoreIOBackend, "is_available", lambda: backend == "ais")
    canonical = "ais" if scheme == "ais" else "s3"
    if backend == "ais":
        bucket = MagicMock()
        bucket.list_all_objects.return_value = [SimpleNamespace(name=k) for k in keys]
        client = MagicMock()
        client.bucket.return_value = bucket
        monkeypatch.setattr(
            "lhotse.serialization.get_aistore_client", lambda: (client, None)
        )
    else:
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {"Contents": [{"Key": key} for key in keys[:3]]},
            {},
            {"Contents": [{"Key": key} for key in keys[3:]]},
        ]
        client = MagicMock()
        client.get_paginator.return_value = paginator
        monkeypatch.setitem(
            sys.modules, "boto3", SimpleNamespace(client=lambda service: client)
        )
    directory = f"{scheme}://bucket/{prefix}"
    fields, streams = _discover_fields(directory)
    assert fields == {"audio", "audio_extra"}
    assert streams == {
        field: [
            f"{canonical}://bucket/{prefix}{field}.{i:06d}.{suffix}" for i in range(2)
        ]
        for field, suffix in [
            ("cuts", "jsonl.gz"),
            ("audio", "tar"),
            ("audio_extra", "tar"),
        ]
    }
    assert LazySharIterator(in_dir=directory).streams == streams
    if backend == "ais":
        client.bucket.assert_called_with("bucket", canonical)
        bucket.list_all_objects.assert_called_with(prefix=prefix)
    else:
        paginator.paginate.assert_called_with(
            Bucket="bucket", Prefix=prefix, Delimiter="/"
        )


def test_local_discovery_preserves_paths_and_excludes_indexes(tmp_path):
    for name in [
        "cuts.000000.jsonl",
        "audio.000000.tar",
        "audio_extra.000000.tar",
        "cuts.000000.jsonl.idx",
        "cuts.000000.jsonl.gzidx",
        "data.idxpack",
    ]:
        (tmp_path / name).touch()
    assert _discover_fields(tmp_path) == (
        {"audio", "audio_extra"},
        {
            field: [tmp_path / f"{field}.000000.{suffix}"]
            for field, suffix in [
                ("cuts", "jsonl"),
                ("audio", "tar"),
                ("audio_extra", "tar"),
            ]
        },
    )


@pytest.mark.parametrize(
    "uri", ["gs://bucket/shar", "s3:///shar", "s3://bucket/shar?query"]
)
def test_invalid_directory_uri(uri):
    with pytest.raises(ValueError):
        _discover_fields(uri)
    with pytest.raises(ValueError):
        LazySharIterator(in_dir=uri)


def test_ais_missing_configuration_does_not_use_s3(monkeypatch):
    monkeypatch.delenv("AIS_ENDPOINT", raising=False)
    from lhotse.serialization import get_aistore_client, is_aistore_available

    get_aistore_client.cache_clear()
    is_aistore_available.cache_clear()
    monkeypatch.setattr("lhotse.serialization.is_module_available", lambda _: True)
    with pytest.raises(ValueError, match="AIS_ENDPOINT"):
        _discover_fields("ais://bucket/shar")
    get_aistore_client.cache_clear()
    is_aistore_available.cache_clear()


def test_short_field_reports_shard_count():
    with pytest.raises(AssertionError, match="Expected 2 shards"):
        LazySharIterator(fields={"cuts": ["a", "b"], "audio": ["x"]})
