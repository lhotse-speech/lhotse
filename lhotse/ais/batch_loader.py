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

    Pass ``force_individual=True`` to skip the MOSS GetBatch call entirely and
    instead issue one AIStore ``Object.get_reader().read_all()`` per object
    (handling tar-member extraction via ``ArchiveConfig(archpath=…)``). This is
    the same per-object path the loader already takes when ``batch.get()``
    raises ``AISError`` or the batch stream truncates — exposed here so callers
    can opt out of MOSS GetBatch up-front when the deployment doesn't support
    it or its performance is degraded for the access pattern.

    Shar lazy-pointer (``shar_ptr``) sources transparently fall back to
    per-object byte-range ``get_reader(byte_range=…)`` calls when the installed
    AIStore SDK / cluster doesn't support byte-range entries in
    MOSS GetBatch requests (or when ``force_individual=True``). The MOSS
    GetBatch path is preferred when available; the byte-range fallback exists
    so non-gzipped lhotse-shar cuts on AIS work even on older deployments.
    """

    def __init__(
        self,
        force_individual: bool = False,
        skip_failed_fetches: bool = False,
    ) -> None:
        """Initialize the AISBatchLoader.

        Construction itself is a no-op w.r.t. AIStore — neither the ``aistore``
        package nor a valid ``AIS_ENDPOINT`` is required here. Both are checked
        lazily on the first AIS access (see :pyattr:`client`), so a CutSet that
        doesn't reference AIStore can flow through :meth:`__call__` unchanged
        even on hosts where AIS isn't configured.

        :param force_individual: when True, every minibatch fetch goes through
            the per-object Get path, never attempting MOSS GetBatch.
        :param skip_failed_fetches: when True, per-object fetch failures (404,
            connection refused, etc.) drop the corresponding cut from the
            returned CutSet instead of raising AISBatchLoaderError. Use as a
            safety net against transient AIS issues or per-shard data
            corruption — every drop is logged at WARNING level so the
            failure rate is observable.
        """
        self.force_individual = force_individual
        self.skip_failed_fetches = skip_failed_fetches
        self._client = None

    @property
    def client(self):
        """Lazily resolve the AIStore client on first access.

        Deferring this lets AISBatchLoader be instantiated unconditionally
        (e.g. by ``AudioSamples(use_batch_loader=True)``) even when the
        current data blend never touches AIS — the ``aistore`` import and
        ``AIS_ENDPOINT`` checks only fire if a CutSet actually contains
        AIS-backed manifests.
        """
        if self._client is None:
            if not is_module_available("aistore"):
                raise ImportError(
                    "Please run 'pip install aistore>=1.17.0' to use AISBatchLoader."
                )
            self._client, _ = get_aistore_client()
        return self._client

    @staticmethod
    def _moss_attrs(info: Any) -> tuple[str, str, str, Optional[str]]:
        """
        Normalise an AIStore batch request/result entry into ``(bck, provider,
        obj_name, archpath)``.

        Verified empirically against ``aistore==1.23.0`` and ``1.25.0``:
          * ``Batch.requests_list`` items are ``aistore.sdk.batch.types.MossIn``
            with the original short attribute names (``bck``, ``provider``,
            ``obj_name``, ``archpath``).
          * ``Batch.get()`` yields ``(MossOut, content)`` tuples and
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
        It handles archive extraction if an archpath is specified, and preserves
        ``start`` / ``length`` byte-range requests used by Shar lazy pointers.

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
        bck, provider, obj_name, archpath = self._moss_attrs(moss_in)
        start, length = self._moss_range(moss_in)

        obj = self.client.bucket(bck_name=bck, provider=provider).object(obj_name)

        if start is not None or length is not None:
            if archpath:
                raise AISBatchLoaderError(
                    "Cannot fall back to direct GET for a request that combines "
                    f"byte range and archive extraction: {obj_name}/{archpath}"
                )
            if start is None or length is None:
                raise AISBatchLoaderError(
                    f"Invalid byte-range request for {obj_name}: "
                    f"start={start!r}, length={length!r}"
                )
            if length <= 0:
                return b""
            end_inclusive = start + length - 1
            return obj.get_reader(
                byte_range=f"bytes={start}-{end_inclusive}"
            ).read_all()

        from aistore.sdk.archive_config import ArchiveConfig

        config = None
        if archpath:
            config = ArchiveConfig(archpath=archpath)

        reader = obj.get_reader(archive_config=config)
        return reader.read_all()

    @staticmethod
    def _moss_range(info: Any) -> tuple[Optional[int], Optional[int]]:
        """Return ``(start, length)`` byte-range fields from a MOSS request."""
        from numbers import Integral

        def _int_or_none(name: str) -> Optional[int]:
            value = getattr(info, name, None)
            if isinstance(value, Integral) and not isinstance(value, bool):
                return int(value)
            return None

        return _int_or_none("start"), _int_or_none("length")

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

        # Pre-scan: if no manifest in this batch references AIStore, skip the
        # entire fetch path. This makes AISBatchLoader a no-op on non-AIS data
        # blends and lets it coexist with hosts where ``aistore`` isn't
        # installed or ``AIS_ENDPOINT`` is unset — the lazy client init below
        # only fires when there's actually AIS data to fetch.
        if not self._cuts_have_ais_data(cuts):
            return cuts

        # Try to use Colocation if available (aistore >= 1.20.0)
        # Colocation optimizes batch requests by grouping objects from the same storage target,
        # reducing network overhead and improving throughput for distributed data retrieval.
        try:
            from aistore.sdk.enums import Colocation

            batch = self.client.batch(colocation=Colocation.TARGET_AWARE)
        except (ImportError, TypeError):
            # Fall back to creating batch without colocation parameter for older versions
            batch = self.client.batch()

        # Decide once per call whether shar_ptr entries can go through MOSS
        # GetBatch byte-range adds. If ``force_individual`` is on, or the SDK
        # predates byte-range MOSS support, route shar_ptr through the per-object
        # byte-range fallback collected in ``shar_ptr_fallback``.
        shar_ptr_uses_batch = (
            not self.force_individual
        ) and self._aistore_byte_range_supported()

        # Per-call fallback queue: each entry is
        # ``(manifest_idx_in_manifest_list, bck, provider, obj_name, offset, length)``.
        # Drained below via per-object ``Object.get_reader(byte_range=…)``.
        shar_ptr_fallback: list[tuple[int, str, str, str, int, int]] = []

        # Collect all URLs for get-batch and track which manifests have URLs
        # plus a parallel list of (cut_idx, has_url) so we can drop entire
        # cuts whose manifests failed to fetch when ``skip_failed_fetches``
        # is on.
        manifest_list = []
        manifest_cut_idx: list[int] = []
        for cut_idx, cut in enumerate(cuts):
            for _, manifest in cut.iter_data():
                manifest_idx = len(manifest_list)
                has_url = self._collect_manifest_urls(
                    manifest,
                    batch,
                    shar_ptr_uses_batch=shar_ptr_uses_batch,
                    shar_ptr_fallback=shar_ptr_fallback,
                    manifest_idx=manifest_idx,
                )
                manifest_list.append((manifest, has_url))
                manifest_cut_idx.append(cut_idx)

        # Execute batch request
        from aistore.sdk.errors import AISError

        # No AIS-backed objects in this CutSet (e.g. all data lives on a non-AIS
        # filesystem). Skip the batch call entirely to avoid spurious warnings.
        if not any(has_url for _, has_url in manifest_list):
            return cuts

        # Index map for the fallback path: manifest_idx → content (or None on
        # failure when ``skip_failed_fetches`` is on). Built before draining the
        # batch so the main injection loop dispatches between the two sources
        # by simple membership check.
        shar_ptr_content: dict[int, Optional[bytes]] = {}
        if shar_ptr_fallback:
            for (
                m_idx,
                bck_,
                provider_,
                obj_name_,
                offset,
                length,
            ) in shar_ptr_fallback:
                end_inclusive = offset + length - 1
                try:
                    reader = (
                        self.client.bucket(bck_name=bck_, provider=provider_)
                        .object(obj_name_)
                        .get_reader(byte_range=f"bytes={offset}-{end_inclusive}")
                    )
                    shar_ptr_content[m_idx] = reader.read_all()
                except Exception as ex:
                    logger.error(
                        f"Byte-range fallback failed for {obj_name_} "
                        f"[{offset}-{end_inclusive}] from bucket "
                        f"{provider_}://{bck_}: {ex}"
                    )
                    if self.skip_failed_fetches:
                        shar_ptr_content[m_idx] = None
                    else:
                        raise AISBatchLoaderError(
                            f"Byte-range fallback failed for {obj_name_} "
                            f"[{offset}-{end_inclusive}]"
                        ) from ex

        # Save requests list before calling batch.get() - it may be cleared
        # after execution. Older/mocked Batch objects may not expose a useful
        # requests_list; the normal MOSS path can still stream from batch.get(),
        # but direct-GET fallback requires this metadata.
        try:
            saved_requests_list = list(batch.requests_list)
        except (AttributeError, TypeError):
            saved_requests_list = []

        moss_manifest_indices = {
            mi
            for mi, (_, has_url) in enumerate(manifest_list)
            if has_url and mi not in shar_ptr_content
        }

        # Generator that mimics ``batch.get()``'s ``(moss_in, content)`` yield
        # contract by issuing one per-object Get instead. Used both as an
        # opt-in primary path (``force_individual=True``) and as the
        # AISError / truncation fallback below.
        #
        # When ``skip_failed_fetches`` is set, per-object errors yield
        # ``(moss_in, None)`` instead of raising; the main loop below treats
        # ``None`` as "drop this manifest's cut" and the affected cuts are
        # filtered out before return. Without the flag, behavior is unchanged
        # (raise AISBatchLoaderError on first failure).
        def _individual_get():
            if not saved_requests_list:
                raise AISBatchLoaderError(
                    "Cannot fall back to sequential AIStore GET: "
                    "batch.requests_list is unavailable or empty."
                )
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

        if not moss_manifest_indices:
            # Only shar_ptr fallback entries — nothing to ask MOSS for.
            batch_result = iter(())
        elif self.force_individual:
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
                # shar_ptr entries that took the byte-range fallback don't
                # consume from ``batch_result``; their content is already in
                # ``shar_ptr_content``.
                if mi in shar_ptr_content:
                    content = shar_ptr_content[mi]
                    if content is None:
                        failed_cut_indices.add(manifest_cut_idx[mi])
                    else:
                        self._inject_data_into_manifest(manifest, content)
                    continue
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
                    direct_info = (
                        saved_requests_list[request_idx]
                        if request_idx < len(saved_requests_list)
                        else info
                    )
                    bck_, provider_, obj_name_, archpath_ = self._moss_attrs(
                        direct_info
                    )
                    logger.warning(
                        f"Object {obj_name_}/{archpath_} from bucket {provider_}://{bck_} "
                        f"returned empty content. Retrying with direct AIStore API call."
                    )
                    try:
                        content = self._get_object_from_moss_in(direct_info)
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

    @staticmethod
    def _cuts_have_ais_data(cuts: CutSet) -> bool:
        """Return True iff any manifest in ``cuts`` is served from AIStore.

        Mirrors the detection conditions in :meth:`_collect_manifest_urls` but
        without touching ``self.client`` or a ``Batch`` — used to short-circuit
        :meth:`__call__` when the CutSet has no AIS-backed data, so
        loaders constructed in environments where AIStore isn't configured
        still pass cuts through unchanged.
        """
        from lhotse.shar.lazy_pointer import decode_pointer

        def _shar_ptr_is_url(pointer: str) -> bool:
            tar_path, _, _ = decode_pointer(pointer)
            return is_valid_url(tar_path)

        for cut in cuts:
            for _, manifest in cut.iter_data():
                if isinstance(manifest, Recording):
                    for source in manifest.sources:
                        if source.type == "url" and is_valid_url(source.source):
                            return True
                        if source.type == "shar_ptr" and _shar_ptr_is_url(
                            source.source
                        ):
                            return True
                elif isinstance(manifest, TemporalArray):
                    inner = manifest.array
                    if inner.storage_type == "shar_ptr_array":
                        if _shar_ptr_is_url(inner.storage_key):
                            return True
                    elif is_valid_url(f"{inner.storage_path}/{inner.storage_key}"):
                        return True
                elif isinstance(manifest, (Array, Features, Image)):
                    if manifest.storage_type == "shar_ptr_array":
                        if _shar_ptr_is_url(manifest.storage_key):
                            return True
                    elif is_valid_url(
                        f"{manifest.storage_path}/{manifest.storage_key}"
                    ):
                        return True
        return False

    def _collect_manifest_urls(
        self,
        manifest: Any,
        batch: Any,
        *,
        shar_ptr_uses_batch: bool,
        shar_ptr_fallback: list,
        manifest_idx: int,
    ) -> bool:
        """
        Add all URLs referenced in a manifest to the batch.

        ``shar_ptr`` entries are routed to ``batch`` when the SDK supports
        byte-range adds (``shar_ptr_uses_batch=True``); otherwise the request
        is queued in ``shar_ptr_fallback`` so :meth:`__call__` can drain it via
        per-object byte-range gets. Either way the manifest is reported as
        having a URL (return value ``True``).

        Returns:
            True if the manifest's data was scheduled for fetch (batch or
            fallback), False otherwise.
        """
        if isinstance(manifest, Recording):
            for source in manifest.sources:
                if source.type == "url":
                    self._add_url_to_batch(source.source, batch)
                    return True
                if source.type == "shar_ptr":
                    if self._add_shar_ptr_to_batch(
                        source.source,
                        batch,
                        shar_ptr_uses_batch=shar_ptr_uses_batch,
                        shar_ptr_fallback=shar_ptr_fallback,
                        manifest_idx=manifest_idx,
                    ):
                        return True
            return False

        elif isinstance(manifest, TemporalArray):
            # TemporalArray wraps an Array, so we need to access the inner array
            inner_array = manifest.array
            if inner_array.storage_type == "shar_ptr_array":
                if self._add_shar_ptr_to_batch(
                    inner_array.storage_key,
                    batch,
                    shar_ptr_uses_batch=shar_ptr_uses_batch,
                    shar_ptr_fallback=shar_ptr_fallback,
                    manifest_idx=manifest_idx,
                ):
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
                if self._add_shar_ptr_to_batch(
                    manifest.storage_key,
                    batch,
                    shar_ptr_uses_batch=shar_ptr_uses_batch,
                    shar_ptr_fallback=shar_ptr_fallback,
                    manifest_idx=manifest_idx,
                ):
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
        Detect whether the installed aistore SDK/cluster generation supports
        byte-range MOSS entries.

        ``aistore==1.25.0`` removed the older ``BatchRequest`` class and still
        has ``Batch.add(..., start=, length=)`` guarded by ``NotImplementedError``.
        The supported API is the lower-level MOSS request schema:
        ``Batch.requests_list`` exposes ``MossReq.moss_in`` and ``MossIn``
        serializes ``start`` / ``length`` in the GetBatch JSON body. Older SDKs
        had partial client-side fields before server support existed, so keep a
        conservative version gate and schema check here.
        """
        try:
            import re

            import aistore
            from aistore.sdk.batch.batch import Batch
            from aistore.sdk.batch.types import MossIn, MossReq
        except Exception:
            return False

        m = re.match(r"^(\d+)\.(\d+)\.(\d+)", getattr(aistore, "__version__", ""))
        if m is None or tuple(map(int, m.groups())) < (1, 25, 0):
            return False

        try:
            descriptor = vars(Batch).get("requests_list")
            if not isinstance(descriptor, property):
                return False
            if "moss_in" not in MossReq.model_fields:
                return False
            if not {"start", "length"}.issubset(MossIn.model_fields):
                return False
            probe = MossIn.model_construct(
                obj_name="__lhotse_probe__.tar",
                bck="__lhotse_probe__",
                provider="ais",
                start=0,
                length=1,
            )
            dumped = probe.model_dump(by_alias=True, exclude_defaults=True)
        except Exception:
            return False

        return dumped.get("start") == 0 and dumped.get("length") == 1

    def _add_shar_ptr_to_batch(
        self,
        pointer: str,
        batch: Any,
        *,
        shar_ptr_uses_batch: bool,
        shar_ptr_fallback: list,
        manifest_idx: int,
    ) -> bool:
        """
        Schedule a Shar lazy pointer fetch.

        When ``shar_ptr_uses_batch`` is True the request is added to the MOSS
        ``Batch`` via direct ``MossIn.model_construct`` append (see
        :meth:`_append_moss_in` for why we bypass ``Batch.add``).
        Otherwise the ``(manifest_idx, bck, provider, obj_name, offset, length)``
        tuple is appended to ``shar_ptr_fallback`` so :meth:`__call__` can
        drain it via per-object byte-range gets.

        Returns True iff the pointer was successfully scheduled (either path).
        """
        from aistore.sdk.utils import parse_url

        from lhotse.shar.lazy_pointer import decode_pointer

        tar_path, offset, end_offset = decode_pointer(pointer)
        if not is_valid_url(tar_path):
            return False
        provider, bck_name, obj_name = parse_url(tar_path)
        if not (provider and bck_name and obj_name):
            return False

        length = end_offset - offset
        if shar_ptr_uses_batch:
            self._append_moss_in(
                batch,
                bck=bck_name,
                provider=provider,
                obj_name=obj_name,
                start=offset,
                length=length,
            )
        else:
            shar_ptr_fallback.append(
                (manifest_idx, bck_name, provider, obj_name, offset, length)
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

        self._append_moss_in(
            batch, bck=bck_name, provider=provider, obj_name=obj_name, archpath=archpath
        )

    def _append_moss_in(
        self,
        batch: Any,
        *,
        bck: str,
        provider: str,
        obj_name: str,
        archpath: Optional[str] = None,
        start: Optional[int] = None,
        length: Optional[int] = None,
    ) -> None:
        """Append one MossIn entry to the batch request, bypassing the SDK's
        ``Batch.add(bucket.object(obj_name), ...)`` path.

        Why this bypass exists
        ----------------------
        ``Batch.add`` builds a fresh ``Bucket`` + ``BucketDetails``
        (Pydantic v2) + ``Object`` + ``MossIn`` (Pydantic v2 with field
        aliases) per call. With ~45 manifests per minibatch in a Granary
        blend, profiling (nsys 2026-05-15, NVTX scope ``ais.collect_urls``)
        showed this loop spends **~1.58 s mean / 4.31 s max per batch on
        pure CPU**, which is ~2/3 of the AIS GetBatch wall time and the
        single biggest hotspot in the worker.

        ``MossIn.model_construct(...)`` skips validation entirely — the
        downstream HTTP serialization only consumes the field values.
        Construction reduces to one dict write into ``batch.request.moss_in``,
        which empirically drops ``ais.collect_urls`` by ~20-50×.

        Risks
        -----
        - Skipping Pydantic validation means a future ``aistore`` SDK that
          adds a required ``MossIn`` field will silently produce invalid
          requests. Pinned to ``aistore<2.0`` at the call site below; the
          unit test in ``test/test_ais_batch_loader_collect_urls.py``
          round-trips ``model_construct`` vs the validating constructor
          and asserts ``model_dump`` equality.
        - ``batch.request.moss_in`` is a non-public attribute. Stable
          through 1.20.0 → 1.25.0; bumping the SDK major version requires
          re-verifying that the field still exists with the same shape.
        """
        # Local imports kept local: aistore is an optional dependency and
        # the module top-level is intentionally aistore-free so AISBatchLoader
        # can be constructed on hosts without the SDK installed (see
        # :pyattr:`client` doc and ``_cuts_have_ais_data`` short-circuit).
        from aistore.sdk.batch.types import MossIn

        # Construct kwargs sparsely so optional fields that the SDK's MossIn
        # schema may not accept (e.g. older versions without start/length)
        # don't surface as model_construct kwargs.
        kwargs = {"bck": bck, "provider": provider, "obj_name": obj_name}
        if archpath is not None:
            kwargs["archpath"] = archpath
        if start is not None:
            kwargs["start"] = start
        if length is not None:
            kwargs["length"] = length

        # Fast path. ``Batch.requests_list`` is the public property accessor
        # for ``Batch.request.moss_in`` (the underlying ``List[MossIn]``);
        # mutating the returned list is equivalent to mutating the field
        # in-place. If the SDK shape ever drifts (rename, restructure), fall
        # back to ``Batch.add(bucket.object(...))`` so we degrade in
        # performance rather than crash. The fast path is exercised by the
        # unit test which round-trips ``model_construct`` against the
        # validating constructor.
        try:
            requests_list = batch.requests_list
            if isinstance(requests_list, list):
                requests_list.append(MossIn.model_construct(**kwargs))
                return
            raise AttributeError
        except AttributeError:
            pass
        # Fallback: original aistore client path. Reaches client.bucket and
        # bucket.object, both of which validate; this is the pre-optimization
        # behavior preserved for forward compatibility and for tests/mocks
        # that don't provide a real Batch.requests_list list.
        if not hasattr(self, "_client"):
            batch.add(obj_name, start=start, length=length, archpath=archpath)
            return
        bucket = self.client.bucket(bck, provider)
        if start is not None or length is not None:
            batch.add(bucket.object(obj_name), start=start, length=length)
        else:
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
                inner_array.storage_type = FILE_TO_MEMORY_TYPE[inner_array.storage_type]
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
