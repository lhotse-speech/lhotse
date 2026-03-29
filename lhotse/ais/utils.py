from typing import Any, Iterable, List, Tuple

from lhotse.serialization import get_aistore_client
from lhotse.utils import Pathlike, split_object_store_url, top_level_object_store_uris


def list_aistore_objects(url: Pathlike) -> List[str]:
    scheme, bucket, prefix = split_object_store_url(url)
    listing = _list_bucket_objects(scheme, bucket, prefix)
    keys = _iter_aistore_object_keys(listing)
    return top_level_object_store_uris(scheme, bucket, prefix, keys)


def _list_bucket_objects(provider: str, bucket: str, prefix: str) -> Any:
    client, _ = get_aistore_client()
    bucket_handle = client.bucket(bucket, provider)
    listing_prefix = f"{prefix}/" if prefix else ""

    for method_name in ("list_all_objects", "list_objects"):
        method = getattr(bucket_handle, method_name, None)
        if method is None:
            continue
        for kwargs in (
            {"prefix": listing_prefix},
            {"prefix_filter": listing_prefix},
        ):
            try:
                return method(**kwargs)
            except TypeError:
                continue

    raise RuntimeError(
        "The installed AIStore SDK does not expose a supported object listing API for Shar directory scanning."
    )


def _iter_aistore_object_keys(listing: Any) -> Iterable[str]:
    entries = getattr(listing, "entries", listing)
    for entry in entries:
        if isinstance(entry, str):
            yield entry
            continue
        if isinstance(entry, dict):
            key = entry.get("name")
        else:
            key = getattr(entry, "name", None)
        if isinstance(key, str):
            yield key
