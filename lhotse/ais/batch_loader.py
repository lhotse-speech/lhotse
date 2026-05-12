import io
import logging
from functools import lru_cache
from typing import Any, Optional

from urllib3.exceptions import TimeoutError

# Get a logger instance for this module
logger = logging.getLogger(__name__)

from lhotse.array import Array, TemporalArray
from lhotse.audio.recording import Recording
from lhotse.cut import CutSet
from lhotse.features.base import Features
from lhotse.image import Image
from lhotse.indexing import read_tar_member_at
from lhotse.serialization import get_aistore_client
from lhotse.utils import is_module_available, is_valid_url

# Mapping between Lhotse file storage types and in-memory equivalents.
FILE_TO_MEMORY_TYPE = {
    "numpy_files": "memory_raw",
    "lilcom_files": "memory_lilcom",
    "pillow_files": "memory_pillow",
}

ARCHIVE_EXTENSIONS = (".tar.gz", ".tar", ".tgz")


def _extract_shar_pointer_payload(block: bytes) -> bytes:
    """Extract the data-member payload bytes out of a Shar lazy-pointer
    byte-range response (``[offset, end_offset)`` of the underlying tar)."""
    data, _path, _info = read_tar_member_at(io.BytesIO(block), 0)
    return data if data is not None else b""


def _shar_ptr_payload_memory_type(payload: bytes) -> str:
    """Pick the in-memory storage_type for a Shar-pointer Array payload."""
    return "memory_npy" if payload[:6] == b"\x93NUMPY" else "memory_lilcom"


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

    Pass ``prefer_individual=True`` to skip the MOSS GetBatch call entirely and
    instead issue one AIStore ``Object.get_reader().read_all()`` per object
    (handling tar-member extraction via ``ArchiveConfig(archpath=…)``). This is
    the same per-object path the loader already takes when ``batch.get()``
    raises ``AISError`` or the batch stream truncates — exposed here so callers
    can opt out of MOSS GetBatch up-front when the deployment doesn't support
    it or its performance is degraded for the access pattern.
    """

    def __init__(
        self,
        prefer_individual: bool = False,
        skip_failed_fetches: bool = False,
    ) -> None:
        """Initialize the AISBatchLoader with an AIStore client and batch context.

        :param prefer_individual: when True, every minibatch fetch goes through
            the per-object Get path, never attempting MOSS GetBatch.
        :param skip_failed_fetches: when True, per-object fetch failures (404,
            connection refused, etc.) drop the corresponding cut from the
            returned CutSet instead of raising AISBatchLoaderError. Use as a
            safety net against transient AIS issues or per-shard data
            corruption — every drop is logged at WARNING level so the
            failure rate is observable.
        """
        if not is_module_available("aistore"):
            raise ImportError(
                "Please run 'pip install aistore>=1.17.0' to use AISBatchLoader."
            )
        self.client, _ = get_aistore_client()
        self.prefer_individual = prefer_individual
        self.skip_failed_fetches = skip_failed_fetches

    @staticmethod
    def _moss_attrs(info: Any) -> tuple[str, str, str, Optional[str]]:
        """
        Normalise an AIStore batch request/result entry into ``(bck, provider,
        obj_name, archpath)``.

        Verified empirically against ``aistore==1.23.0``:
          * ``BatchRequest.requests_list`` items are ``aistore.sdk.batch.types.MossIn``
            with the original short attribute names (``bck``, ``provider``,
            ``obj_name``, ``archpath``).
          * ``BatchRequest.get()`` yields ``(MossOut, content)`` tuples and
            ``MossOut`` carries different attribute names. To stay robust
            against further SDK churn, this helper falls back through every
            naming convention we've seen in the wild
            (``bck`` / ``bucket_name``; ``provider`` / ``bucket_provider``;
            ``obj_name`` / ``object_name`` / ``name``).

        Raises ``AttributeError`` only if NONE of the expected names exist —
        a clearer failure than the original ``'MossOut' object has no
        attribute 'bck'`` deep in the retry path.
        """
        def _first(o, *names):
            for n in names:
                v = getattr(o, n, None)
                if v is not None:
                    return v
            raise AttributeError(
                f"{type(o).__name__} object has none of the expected attributes "
                f"{names!r} — AIStore SDK API may have changed again; "
                f"available attrs: {[a for a in dir(o) if not a.startswith('_')][:20]}"
            )
        bck = _first(info, "bck", "bucket_name", "bucket")
        provider = _first(info, "provider", "bucket_provider")
        obj_name = _first(info, "obj_name", "object_name", "name")
        archpath = getattr(info, "archpath", None)
        return bck, provider, obj_name, archpath

    def _get_object_from_moss_in(self, moss_in: Any) -> bytes:
        """
        Fetch a single object from AIStore using the ObjectNames request info.

        This method is used as a fallback when batch operations fail or return empty content.
        It handles archive extraction if an archpath is specified.

        Args:
            moss_in: AIStore ObjectNames request — accepts both ``MossIn`` (from
                ``batch.requests_list``) and ``MossOut`` (from ``batch.get()``
                iterator) shapes; attribute access goes through
                :meth:`_moss_attrs` which handles both.

        Returns:
            The object content as bytes.

        Raises:
            Exception: If the object cannot be fetched from AIStore.
        """
        from aistore.sdk.archive_config import ArchiveConfig

        bck, provider, obj_name, archpath = self._moss_attrs(moss_in)

        config = None
        if archpath:
            config = ArchiveConfig(archpath=archpath)

        reader = (
            self.client.bucket(bck_name=bck, provider=provider)
            .object(obj_name)
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

        # Try to use Colocation if available (aistore >= 1.20.0)
        # Colocation optimizes batch requests by grouping objects from the same storage target,
        # reducing network overhead and improving throughput for distributed data retrieval.
        try:
            from aistore.sdk.enums import Colocation

            batch = self.client.batch(colocation=Colocation.TARGET_AWARE)
        except (ImportError, TypeError):
            # Fall back to creating batch without colocation parameter for older versions
            batch = self.client.batch()
        # Collect all URLs for get-batch and track which manifests have URLs
        # plus a parallel list of (cut_idx, has_url) so we can drop entire
        # cuts whose manifests failed to fetch when ``skip_failed_fetches``
        # is on.
        manifest_list = []
        manifest_cut_idx: list[int] = []
        for cut_idx, cut in enumerate(cuts):
            for _, manifest in cut.iter_data():
                has_url = self._collect_manifest_urls(manifest, batch)
                manifest_list.append((manifest, has_url))
                manifest_cut_idx.append(cut_idx)

        # Execute batch request
        from aistore.sdk.errors import AISError

        # No AIS-backed objects in this CutSet (e.g. all data lives on a non-AIS
        # filesystem). Skip the batch call entirely to avoid spurious warnings.
        if not any(has_url for _, has_url in manifest_list):
            return cuts

        # Save requests list before calling batch.get() - it may be cleared after execution
        saved_requests_list = list(batch.requests_list)

        # Generator that mimics ``batch.get()``'s ``(moss_in, content)`` yield
        # contract by issuing one per-object Get instead. Used both as an
        # opt-in primary path (``prefer_individual=True``) and as the
        # AISError / truncation fallback below.
        #
        # When ``skip_failed_fetches`` is set, per-object errors yield
        # ``(moss_in, None)`` instead of raising; the main loop below treats
        # ``None`` as "drop this manifest's cut" and the affected cuts are
        # filtered out before return. Without the flag, behavior is unchanged
        # (raise AISBatchLoaderError on first failure).
        def _individual_get():
            for moss_in in saved_requests_list:
                try:
                    content = self._get_object_from_moss_in(moss_in)
                    yield (moss_in, content)
                except Exception as ex:
                    bck_, provider_, obj_name_, archpath_ = self._moss_attrs(moss_in)
                    logger.error(
                        f"Failed to fetch object {obj_name_}"
                        + (f" archpath={archpath_}" if archpath_ else "")
                        + f" from bucket {provider_}://{bck_}: {ex}"
                    )
                    if self.skip_failed_fetches:
                        yield (moss_in, None)
                    else:
                        raise AISBatchLoaderError(
                            f"Sequential GET fallback failed for {obj_name_}"
                        ) from ex

        if self.prefer_individual:
            batch_result = _individual_get()
        else:
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
                batch_result = _individual_get()

        # Apply the received data back into each manifest that had a URL.
        # ``failed_cut_indices`` accumulates cuts whose manifests failed to
        # fetch when ``skip_failed_fetches`` is on; they are removed from the
        # returned CutSet (instead of raising).
        failed_cut_indices: set[int] = set()
        request_idx = 0
        batch_stream_failed = False
        for mi, (manifest, has_url) in enumerate(manifest_list):
            if has_url:
                info = None
                content = None

                if batch_stream_failed:
                    # Batch stream already broke — go straight to individual GET
                    info = saved_requests_list[request_idx]
                    content = b""  # trigger retry below
                else:
                    try:
                        info, content = next(batch_result)
                    except StopIteration:
                        # Batch stream was truncated (e.g., connection reset mid-tar).
                        # Fall back to individual GET for this and all remaining objects.
                        batch_stream_failed = True
                        logger.warning(
                            f"Batch stream truncated at index {request_idx}/{len(saved_requests_list)}. "
                            f"Falling back to direct AIStore API calls for remaining objects."
                        )
                        if request_idx < len(saved_requests_list):
                            info = saved_requests_list[request_idx]
                            content = b""  # trigger retry below
                        else:
                            raise AISBatchLoaderError(
                                f"Batch stream truncated at index {request_idx}, but cannot recover: "
                                f"index out of range for saved_requests_list (len={len(saved_requests_list)})"
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
                    bck_, provider_, obj_name_, archpath_ = self._moss_attrs(info)
                    logger.warning(
                        f"Object {obj_name_}/{archpath_} from bucket {provider_}://{bck_} "
                        f"returned empty content. Retrying with direct AIStore API call."
                    )
                    try:
                        content = self._get_object_from_moss_in(info)
                    except Exception as ex:
                        logger.error(
                            f"Failed to fetch object {obj_name_} from bucket "
                            f"{provider_}://{bck_}: {ex}"
                        )
                        if self.skip_failed_fetches:
                            content = None  # signal "drop this cut"
                        else:
                            raise AISBatchLoaderError(
                                f"Direct API fallback failed for {obj_name_}"
                            ) from ex

                if content is None:
                    # Per-object fetch failed and skip_failed_fetches is on:
                    # mark the parent cut for removal and skip injection.
                    failed_cut_indices.add(manifest_cut_idx[mi])
                else:
                    self._inject_data_into_manifest(manifest, content)
                request_idx += 1

        if failed_cut_indices:
            survivors = [c for i, c in enumerate(cuts) if i not in failed_cut_indices]
            logger.warning(
                f"AISBatchLoader dropping {len(failed_cut_indices)}/{len(cuts)} cuts "
                f"due to per-object fetch failures (skip_failed_fetches=True)."
            )
            return CutSet(survivors)

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
                if source.type == "shar_ptr":
                    # Forward-compat scaffold: when aistore SDK exposes
                    # byte-range support in BatchRequest.add(start=, length=),
                    # route through ``_add_shar_ptr_to_batch``. Until then,
                    # the loader falls back to per-cut _prepare_for_reading.
                    if self._add_shar_ptr_to_batch(source.source, batch):
                        return True
            return False

        elif isinstance(manifest, TemporalArray):
            # TemporalArray wraps an Array, so we need to access the inner array
            inner_array = manifest.array
            if inner_array.storage_type == "shar_ptr_array":
                if self._add_shar_ptr_to_batch(inner_array.storage_key, batch):
                    return True
                return False
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
            if manifest.storage_type == "shar_ptr_array":
                if self._add_shar_ptr_to_batch(manifest.storage_key, batch):
                    return True
                return False
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

    @staticmethod
    @lru_cache(maxsize=1)
    def _aistore_byte_range_supported() -> bool:
        """
        Detect whether the installed aistore SDK accepts byte-range fetch in
        :meth:`aistore.sdk.batch.batch.BatchRequest.add` *without raising*.

        Probe: instantiate a ``BatchRequest`` and call ``add(start=0,
        length=0)`` against a sentinel object. If the SDK validates byte-range
        usage eagerly with ``NotImplementedError`` (current behaviour, see
        ``aistore/sdk/batch/batch.py``), this fails locally before any IO.
        Cached for the process lifetime.
        """
        try:
            from aistore.sdk.batch.batch import BatchRequest
        except Exception:
            return False
        try:
            req = BatchRequest()
            req.add(object(), start=0, length=0)
        except NotImplementedError:
            return False
        except TypeError:
            # ``start`` / ``length`` not in the signature on older SDKs.
            return False
        except Exception:
            # Any other exception (e.g. invalid object stub) means the SDK
            # got past byte-range validation, so the feature is supported.
            return True
        return True

    def _add_shar_ptr_to_batch(self, pointer: str, batch: Any) -> bool:
        """
        Add a Shar lazy pointer to an in-flight AIS batch using byte-range
        fetch. Returns True iff the pointer was successfully scheduled
        (requires the aistore SDK to support byte-range batch).
        """
        if not self._aistore_byte_range_supported():
            return False
        from aistore.sdk.utils import parse_url

        from lhotse.shar.lazy_pointer import decode_pointer

        tar_path, offset, end_offset = decode_pointer(pointer)
        if not is_valid_url(tar_path):
            return False
        provider, bck_name, obj_name = parse_url(tar_path)
        if not (provider and bck_name and obj_name):
            return False
        bucket = self.client.bucket(bck_name, provider)
        batch.add(
            bucket.object(obj_name), start=offset, length=end_offset - offset
        )
        return True

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
                if source.type == "shar_ptr":
                    content = _extract_shar_pointer_payload(content)
                source.type = "memory"
                source.source = content

        elif isinstance(manifest, TemporalArray):
            # TemporalArray wraps an Array, so update the inner array
            inner_array = manifest.array
            if inner_array.storage_type == "shar_ptr_array":
                payload = _extract_shar_pointer_payload(content)
                inner_array.storage_type = _shar_ptr_payload_memory_type(payload)
                inner_array.storage_path = ""
                inner_array.storage_key = payload
            else:
                inner_array.storage_type = FILE_TO_MEMORY_TYPE[
                    inner_array.storage_type
                ]
                inner_array.storage_path = ""
                inner_array.storage_key = content

        elif isinstance(manifest, (Array, Features, Image)):
            if manifest.storage_type == "shar_ptr_array":
                payload = _extract_shar_pointer_payload(content)
                manifest.storage_type = _shar_ptr_payload_memory_type(payload)
                manifest.storage_path = ""
                manifest.storage_key = payload
            else:
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
