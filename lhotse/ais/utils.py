"""List objects in AIStore buckets."""

from typing import Iterable


def list_aistore_objects(provider: str, bucket: str, prefix: str) -> Iterable[str]:
    """List all object names that start with ``prefix``."""
    from lhotse.serialization import get_aistore_client

    client, _ = get_aistore_client()
    return (
        entry.name
        for entry in client.bucket(bucket, provider).list_all_objects(prefix=prefix)
    )
