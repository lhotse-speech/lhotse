"""AIStore object discovery without importing the optional SDK at module load."""

from typing import Iterable


def list_aistore_objects(provider: str, bucket: str, prefix: str) -> Iterable[str]:
    """List all object keys under a prefix, including every listing page."""
    from lhotse.serialization import get_aistore_client

    client, _ = get_aistore_client()
    return (
        entry.name
        for entry in client.bucket(bucket, provider).list_all_objects(prefix=prefix)
    )
