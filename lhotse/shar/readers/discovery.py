"""Find Shar shards in local directories and object stores."""

from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple
from urllib.parse import urlparse

from lhotse.serialization import AIStoreIOBackend, extension_contains
from lhotse.utils import Pathlike


def discover_shar_fields(
    in_dir: Pathlike,
) -> Tuple[Set[str], Dict[str, List[Pathlike]]]:
    """
    Find Shar shards in a local directory or an AIStore or S3 location.

    Returns ``(fields, streams)``. The ``fields`` set contains the data field
    names, such as ``recording`` and ``features``. The ``streams`` dictionary
    maps each field, including ``cuts``, to a sorted list of shard paths.
    Local paths are returned as ``Path`` objects and remote paths as URL strings.
    Index files (``.idx``, ``.gzidx`` and ``.idxpack``) are skipped.

    :param in_dir: directory or ``ais://`` / ``s3://`` location containing Shar shards.
    """
    streams: Dict[str, List[Pathlike]] = {}
    for path in _list_shar_directory(in_dir):
        if Path(path).name.endswith((".idx", ".gzidx", ".idxpack")):
            continue
        field = Path(path).name.split(".")[0]
        if field == "cuts" and not extension_contains(".jsonl", path):
            continue
        streams.setdefault(field, []).append(path)

    assert streams.get("cuts"), f"No cuts JSONL shards found in {in_dir}"
    for paths in streams.values():
        paths.sort()
    return set(streams) - {"cuts"}, streams


def _list_shar_directory(in_dir: Pathlike) -> Iterable[Pathlike]:
    if isinstance(in_dir, Path) or "://" not in in_dir:
        yield from Path(in_dir).glob("*")
        return

    parsed = urlparse(in_dir)
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

        keys = list_aistore_objects(
            provider=scheme, bucket=parsed.netloc, prefix=prefix
        )
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
    for key in keys:
        if key.startswith(prefix) and key != prefix and "/" not in key[len(prefix) :]:
            yield f"{scheme}://{parsed.netloc}/{key}"
