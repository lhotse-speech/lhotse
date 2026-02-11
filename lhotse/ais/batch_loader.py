import logging
from typing import Any, Optional

from urllib3.exceptions import TimeoutError

# Get a logger instance for this module
logger = logging.getLogger(__name__)

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

    def _get_object_from_moss_in(self, moss_in: Any) -> bytes:
        """
        Fetch a single object from AIStore using the ObjectNames request info.

        This method is used as a fallback when batch operations fail or return empty content.
        It handles archive extraction if an archpath is specified.

        Args:
            moss_in: AIStore ObjectNames request containing bucket, object, and optional archpath.

        Returns:
            The object content as bytes.

        Raises:
            Exception: If the object cannot be fetched from AIStore.
        """
        try:
            from aistore.sdk.archive_config import ArchiveConfig
        except ImportError:
            # Older versions may not have ArchiveConfig
            ArchiveConfig = None

        config = None
        if ArchiveConfig and hasattr(moss_in, "archpath") and moss_in.archpath:
            config = ArchiveConfig(archpath=moss_in.archpath)

        reader = (
            self.client.bucket(bck_name=moss_in.bck, provider=moss_in.provider)
            .object(moss_in.obj_name)
            .get_reader(archive_config=config)
        )
        return reader.read_all()

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

        # Try to use Colocation if available (newer versions)
        # Colocation optimizes batch requests by grouping objects from the same storage target,
        # reducing network overhead and improving throughput for distributed data retrieval.
        try:
            from aistore.sdk.enums import Colocation

            batch = self.client.batch(colocation=Colocation.TARGET_AWARE)
        except (ImportError, TypeError):
            # Fall back to creating batch without colocation parameter for older versions
            batch = self.client.batch()
        # Collect all URLs for get-batch and track which manifests have URLs
        manifest_list = []
        for cut in cuts:
            for _, manifest in cut.iter_data():
                has_url = self._collect_manifest_urls(manifest, batch)
                manifest_list.append((manifest, has_url))

        # Execute batch request
        from aistore.sdk.errors import AISError

        # Save requests list before calling batch.get() - it may be cleared after execution
        saved_requests_list = list(batch.requests_list)

        try:
            batch_result = batch.get()
        except ValueError as e:
            # ValueError occurs when the batch request is invalid or empty
            logger.warning(
                f"ValueError during batch.get(): {e}. Returning unmodified cuts."
            )
            return cuts
        except AISError as e:
            logger.warning(
                f"AIStore batch.get() failed: {e}. Falling back to sequential GET requests."
            )
            # Fallback: make sequential GET requests for each object in the batch
            # Use a generator to maintain consistency with batch.get() which returns an iterator
            def sequential_get():
                for moss_in in saved_requests_list:
                    try:
                        content = self._get_object_from_moss_in(moss_in)
                        yield (moss_in, content)
                    except Exception as ex:
                        logger.error(
                            f"Failed to fetch object {moss_in.obj_name} from bucket "
                            f"{moss_in.provider}://{moss_in.bck}: {ex}"
                        )
                        raise AISBatchLoaderError(
                            f"Sequential GET fallback failed for {moss_in.obj_name}"
                        ) from ex

            batch_result = sequential_get()

        # Apply the received data back into each manifest that had a URL
        request_idx = 0
        for manifest, has_url in manifest_list:
            if has_url:
                info = None
                content = None

                try:
                    info, content = next(batch_result)
                except StopIteration:
                    raise AISBatchLoaderError(
                        "Batch result iterator exhausted prematurely. "
                        f"Expected more objects for manifests with URLs."
                    )
                except TimeoutError as e:
                    # Timeout occurred - recover the request info from saved_requests_list
                    logger.warning(
                        f"Timeout while fetching batch result at index {request_idx}: {e}. "
                        f"Falling back to direct AIStore API call."
                    )

                    if request_idx < len(saved_requests_list):
                        info = saved_requests_list[request_idx]
                        content = b""  # Mark as empty to trigger retry
                    else:
                        raise AISBatchLoaderError(
                            f"Timeout at request index {request_idx}, but cannot recover: "
                            f"index out of range for saved_requests_list (len={len(saved_requests_list)})"
                        ) from e

                # Retry with direct API call if content is empty (from timeout or actual empty response)
                if content == b"":
                    logger.warning(
                        f"Object {info.obj_name}/{info.archpath} from bucket {info.provider}://{info.bck} "
                        f"returned empty content. Retrying with direct AIStore API call."
                    )
                    try:
                        content = self._get_object_from_moss_in(info)
                    except Exception as ex:
                        logger.error(
                            f"Failed to fetch object {info.obj_name} from bucket "
                            f"{info.provider}://{info.bck}: {ex}"
                        )
                        raise AISBatchLoaderError(
                            f"Direct API fallback failed for {info.obj_name}"
                        ) from ex

                self._inject_data_into_manifest(manifest, content)
                request_idx += 1

        return cuts

    # ----------------------------- Internal Helpers -----------------------------

    def _collect_manifest_urls(self, manifest: Any, batch: Any) -> bool:
        """
        Add all URLs referenced in a manifest to the batch.

        Returns:
            True if URLs were added to the batch, False otherwise.
        """
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
