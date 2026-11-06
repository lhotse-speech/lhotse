"""Shared directory discovery for streaming, indexed and packed Shar readers."""

from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple
from urllib.parse import urlparse

from lhotse.serialization import AIStoreIOBackend, extension_contains
from lhotse.utils import Pathlike


def _list_shar_directory(in_dir: Pathlike) -> Iterable[Pathlike]:
    # Path objects are always local; do not interpret a colon in a local name.
    parsed = urlparse(in_dir) if isinstance(in_dir, str) and "://" in in_dir else None
    if parsed is None or not parsed.scheme:
        return Path(in_dir).glob("*")
    scheme = parsed.scheme.lower()
    if scheme not in {"ais", "s3", "s3a", "s3n"}:
        raise ValueError(f"Unsupported Shar directory URI scheme: {scheme}")
    if not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError(f"Invalid Shar directory URI: {in_dir}")
    scheme = "s3" if scheme in {"s3a", "s3n"} else scheme
    prefix = parsed.path.lstrip("/").rstrip("/")
    prefix = f"{prefix}/" if prefix else ""
    if scheme == "ais" or AIStoreIOBackend.is_available():
        from lhotse.ais import list_aistore_objects

        keys = list_aistore_objects(scheme, parsed.netloc, prefix)
    else:
        try:
            import boto3
        except ImportError as e:
            raise ImportError(
                "Install boto3 to discover Shar shards in S3: pip install boto3"
            ) from e
        pages = (
            boto3.client("s3")
            .get_paginator("list_objects_v2")
            .paginate(Bucket=parsed.netloc, Prefix=prefix, Delimiter="/")
        )
        keys = (item["Key"] for page in pages for item in page.get("Contents", []))
    return (
        f"{scheme}://{parsed.netloc}/{key}"
        for key in keys
        if key.startswith(prefix) and key != prefix and "/" not in key[len(prefix) :]
    )


def _discover_fields(in_dir: Pathlike) -> Tuple[Set[str], Dict[str, List[Pathlike]]]:
    """Discover shard fields from an ``in_dir``.

    Returns ``(fields, streams)`` where *fields* is a set of non-cuts
    field names and *streams* maps each field (plus ``"cuts"``) to a
    sorted list of shard paths.

    Index files (``.idx``, ``.gzidx`` and ``.idxpack``) are excluded from discovery.
    """
    all_paths = [
        p
        for p in _list_shar_directory(in_dir)
        if not Path(p).name.endswith((".idx", ".gzidx", ".idxpack"))
    ]
    streams: Dict[str, List[Pathlike]] = {}
    for path in all_paths:
        field = Path(path).name.split(".")[0]
        if field != "cuts" or extension_contains(".jsonl", path):
            streams.setdefault(field, []).append(path)
    assert streams.get("cuts"), f"No cuts JSONL shards found in {in_dir}"
    for paths in streams.values():
        paths.sort()
    return set(streams) - {"cuts"}, streams
