from typing import Any, Dict, Iterator, Optional, Tuple

from lhotse.array import Array, TemporalArray
from lhotse.audio.recording import Recording
from lhotse.cut import CutSet
from lhotse.features.base import Features
from lhotse.image import Image
from lhotse.serialization import get_aistore_client
from lhotse.utils import is_module_available, is_valid_url

# Mapping between Lhotse file storage types and in-memory equivalents.
FILE_TO_MEMORY_TYPE = {
    "numpy_files": "memory_raw",
    "lilcom_files": "memory_lilcom",
    "pillow_files": "memory_pillow",
}

ARCHIVE_EXTENSIONS = (".tar.gz", ".tar", ".tgz")


class AISBatchLoaderError(Exception):
    """Base exception for AISBatchLoader operations."""


class AISBatchLoader:
    """
    Loads all data referenced by a :class:`CutSet` in a single AIStore Get-Batch call.

    The loader optimizes I/O by aggregating all object URLs from a CutSet and requesting
    them together. It offloads archive extraction and data slicing to AIStore, avoiding
    redundant downloads and local decompression.

    Example:
        >>> loader = AISBatchLoader()
        >>> cuts_with_data = loader(cuts)
    """

    def __init__(self) -> None:
        """Initialize the AISBatchLoader with an AIStore client and batch context."""
        if not is_module_available("aistore"):
            raise ImportError(
                "Please run 'pip install aistore>=1.17.0' to use AISBatchLoader."
            )
        self.client, _ = get_aistore_client()

    def __call__(self, cuts: CutSet) -> CutSet:
        """
        Fetch all data referenced by a CutSet in one AIStore batch operation.

        Args:
            cuts: A non-lazy CutSet representing a single batch of data.

        Returns:
            The same CutSet object with all manifests updated to reference in-memory data.

        Raises:
            ValueError: If the input CutSet is lazy.
            AISBatchLoaderError: For invalid URLs or unsupported storage types.
        """
        if cuts.is_lazy:
            raise ValueError(
                "Lazy CutSets cannot be used with AISBatchLoader. "
                "Convert to eager via `cuts.to_eager()` before loading."
            )

        batch = self.client.batch()
        # Collect all URLs for get-batch and track which manifests have URLs
        manifest_list = []
        for cut in cuts:
            for _, manifest in cut.iter_data():
                has_url = self._collect_manifest_urls(manifest, batch)
                manifest_list.append((manifest, has_url))

        # Execute batch request
        batch_result = batch.get()

        # Apply the received data back into each manifest that had a URL
        for manifest, has_url in manifest_list:
            if has_url:
                _, content = next(batch_result)
                self._inject_data_into_manifest(manifest, content)

        return cuts

    # ----------------------------- Internal Helpers -----------------------------

    def _collect_manifest_urls(self, manifest: Any, batch: Any) -> None:
        """Add all URLs referenced in a manifest to the batch."""
        if isinstance(manifest, Recording):
            for source in manifest.sources:
                if source.type == "url":
                    self._add_url_to_batch(source.source, batch)
                    return True
            return False

        elif isinstance(manifest, TemporalArray):
            # TemporalArray wraps an Array, so we need to access the inner array
            inner_array = manifest.array
            if inner_array.storage_type not in FILE_TO_MEMORY_TYPE:
                raise AISBatchLoaderError(
                    f"Unsupported storage type '{inner_array.storage_type}'. "
                    f"Supported types: {list(FILE_TO_MEMORY_TYPE.keys())}"
                )

            obj_path = f"{inner_array.storage_path}/{inner_array.storage_key}"
            if is_valid_url(obj_path):
                self._add_url_to_batch(obj_path, batch)
                return True
            return False

        elif isinstance(manifest, (Array, Features, Image)):
            if manifest.storage_type not in FILE_TO_MEMORY_TYPE:
                raise AISBatchLoaderError(
                    f"Unsupported storage type '{manifest.storage_type}'. "
                    f"Supported types: {list(FILE_TO_MEMORY_TYPE.keys())}"
                )

            obj_path = f"{manifest.storage_path}/{manifest.storage_key}"
            if is_valid_url(obj_path):
                self._add_url_to_batch(obj_path, batch)
                return True
            return False

        return False

    def _add_url_to_batch(self, url: str, batch: Any) -> None:
        """Add a single AIStore URL to the batch request."""
        from aistore.sdk.utils import parse_url

        provider, bck_name, obj_name = parse_url(url)
        if not (provider and bck_name and obj_name):
            raise AISBatchLoaderError(f"Invalid object URL: '{url}'")

        arch_ext = self._get_archive_extension(obj_name)
        archpath = None
        if arch_ext and arch_ext in obj_name:
            prefix, _, suffix = obj_name.partition(f"{arch_ext}/")
            obj_name, archpath = prefix + arch_ext, suffix

        bucket = self.client.bucket(bck_name, provider)
        batch.add(bucket.object(obj_name), archpath=archpath)

    def _inject_data_into_manifest(self, manifest: Any, content: bytes) -> None:
        """Replace manifest storage references with in-memory content."""
        if isinstance(manifest, Recording):
            for source in manifest.sources:
                source.type = "memory"
                source.source = content

        elif isinstance(manifest, TemporalArray):
            # TemporalArray wraps an Array, so update the inner array
            inner_array = manifest.array
            inner_array.storage_type = FILE_TO_MEMORY_TYPE[inner_array.storage_type]
            inner_array.storage_path = ""
            inner_array.storage_key = content

        elif isinstance(manifest, (Array, Features, Image)):
            manifest.storage_type = FILE_TO_MEMORY_TYPE[manifest.storage_type]
            manifest.storage_path = ""
            manifest.storage_key = content

    @staticmethod
    def _get_archive_extension(obj_name: str) -> Optional[str]:
        """Return the supported archive extension if present in the object name."""
        for ext in ARCHIVE_EXTENSIONS:
            if ext in obj_name:
                return ext
        return None
