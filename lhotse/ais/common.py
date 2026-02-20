"""
Shared constants, helpers, and base exception for AIStore loaders.
"""

from typing import Any, Optional, Tuple

from lhotse.array import Array, TemporalArray
from lhotse.audio.recording import Recording
from lhotse.features.base import Features
from lhotse.image import Image
from lhotse.utils import is_valid_url

# Mapping between Lhotse file storage types and in-memory equivalents.
FILE_TO_MEMORY_TYPE = {
    "numpy_files": "memory_raw",
    "lilcom_files": "memory_lilcom",
    "pillow_files": "memory_pillow",
}

ARCHIVE_EXTENSIONS = (".tar.gz", ".tar", ".tgz")


class AISLoaderError(Exception):
    """Base exception for all AIStore loader operations."""


def get_archive_extension(obj_name: str) -> Optional[str]:
    """Return the supported archive extension if present in the object name."""
    for ext in ARCHIVE_EXTENSIONS:
        if ext in obj_name:
            return ext
    return None


def parse_ais_url(url: str) -> Tuple[str, str, str, Optional[str]]:
    """
    Parse an AIStore URL into its components, extracting any archive path.

    Returns:
        A tuple of (provider, bck_name, obj_name, archpath).
        archpath is None if no archive extension is found in obj_name.

    Raises:
        AISLoaderError: If the URL cannot be parsed into valid components.
    """
    from aistore.sdk.utils import parse_url

    provider, bck_name, obj_name = parse_url(url)
    if not (provider and bck_name and obj_name):
        raise AISLoaderError(f"Invalid object URL: '{url}'")

    archpath = None
    arch_ext = get_archive_extension(obj_name)
    if arch_ext is not None:
        prefix, _, suffix = obj_name.partition(f"{arch_ext}/")
        obj_name, archpath = prefix + arch_ext, suffix

    return provider, bck_name, obj_name, archpath


def extract_manifest_url(manifest: Any) -> Optional[str]:
    """
    Extract the fetchable URL from a manifest object.

    Supports Recording (url-type sources), Array, Features, Image, and TemporalArray.

    Returns:
        The URL string if the manifest references remote AIS data, None otherwise.

    Raises:
        AISLoaderError: If the manifest has an unsupported storage type.
    """
    if isinstance(manifest, Recording):
        for source in manifest.sources:
            if source.type == "url":
                return source.source
        return None

    elif isinstance(manifest, TemporalArray):
        inner_array = manifest.array
        if inner_array.storage_type not in FILE_TO_MEMORY_TYPE:
            raise AISLoaderError(
                f"Unsupported storage type '{inner_array.storage_type}'. "
                f"Supported types: {list(FILE_TO_MEMORY_TYPE.keys())}"
            )
        obj_path = f"{inner_array.storage_path}/{inner_array.storage_key}"
        return obj_path if is_valid_url(obj_path) else None

    elif isinstance(manifest, (Array, Features, Image)):
        if manifest.storage_type not in FILE_TO_MEMORY_TYPE:
            raise AISLoaderError(
                f"Unsupported storage type '{manifest.storage_type}'. "
                f"Supported types: {list(FILE_TO_MEMORY_TYPE.keys())}"
            )
        obj_path = f"{manifest.storage_path}/{manifest.storage_key}"
        return obj_path if is_valid_url(obj_path) else None

    return None


def inject_data_into_manifest(manifest: Any, content: bytes) -> None:
    """Replace manifest storage references with in-memory content."""
    if isinstance(manifest, Recording):
        for source in manifest.sources:
            source.type = "memory"
            source.source = content

    elif isinstance(manifest, TemporalArray):
        inner_array = manifest.array
        inner_array.storage_type = FILE_TO_MEMORY_TYPE[inner_array.storage_type]
        inner_array.storage_path = ""
        inner_array.storage_key = content

    elif isinstance(manifest, (Array, Features, Image)):
        manifest.storage_type = FILE_TO_MEMORY_TYPE[manifest.storage_type]
        manifest.storage_path = ""
        manifest.storage_key = content
