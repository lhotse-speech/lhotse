import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Tuple

logger = logging.getLogger(__name__)

from lhotse.ais.common import (
    AISLoaderError,
    extract_manifest_url,
    inject_data_into_manifest,
    parse_ais_url,
)
from lhotse.cut import CutSet
from lhotse.serialization import get_aistore_client
from lhotse.utils import is_module_available


class AISObjectLoaderError(AISLoaderError):
    """Exception for AISObjectLoader operations."""


class AISObjectLoader:
    """
    Loads data referenced by a :class:`CutSet` via individual AIStore GET calls.

    Unlike :class:`AISBatchLoader` which uses the batch GET API, this loader
    makes individual GET requests per object. Concurrency is controlled via
    ``max_workers``.

    Example:
        >>> loader = AISObjectLoader(max_workers=4)
        >>> cuts_with_data = loader(cuts)
    """

    def __init__(self, max_workers: int = 1) -> None:
        """
        Initialize the AISObjectLoader.

        Args:
            max_workers: Number of concurrent fetch threads.
                1 (default) uses a simple sequential loop with no thread pool overhead.
                >1 uses a ThreadPoolExecutor for parallel fetches.

        Raises:
            ImportError: If the ``aistore`` package is not installed.
            ValueError: If ``max_workers`` is less than 1.
        """
        if not is_module_available("aistore"):
            raise ImportError(
                "Please run 'pip install aistore>=1.17.0' to use AISObjectLoader."
            )
        if max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {max_workers}")
        self.max_workers = max_workers
        self.client, _ = get_aistore_client()

    def __call__(self, cuts: CutSet) -> CutSet:
        """
        Fetch all data referenced by a CutSet via individual GET requests.

        Args:
            cuts: A non-lazy CutSet representing a single batch of data.

        Returns:
            The same CutSet object with all manifests updated to reference in-memory data.

        Raises:
            ValueError: If the input CutSet is lazy.
            AISObjectLoaderError: For fetch failures or invalid URLs.
        """
        if cuts.is_lazy:
            raise ValueError(
                "Lazy CutSets cannot be used with AISObjectLoader. "
                "Convert to eager via `cuts.to_eager()` before loading."
            )

        # Collect (manifest, url) pairs
        fetch_tasks: List[Tuple[Any, str]] = []
        for cut in cuts:
            for _, manifest in cut.iter_data():
                url = extract_manifest_url(manifest)
                if url is not None:
                    fetch_tasks.append((manifest, url))

        if not fetch_tasks:
            return cuts

        if self.max_workers == 1:
            self._fetch_sequential(fetch_tasks)
        else:
            self._fetch_concurrent(fetch_tasks)

        return cuts

    def _fetch_object(self, url: str) -> bytes:
        """
        Fetch a single object from AIStore.

        Args:
            url: The AIStore URL to fetch.

        Returns:
            The object content as bytes.

        Raises:
            AISObjectLoaderError: If the fetch fails.
        """
        try:
            from aistore.sdk.archive_config import ArchiveConfig
        except ImportError:
            ArchiveConfig = None

        try:
            provider, bck_name, obj_name, archpath = parse_ais_url(url)
        except AISLoaderError as e:
            raise AISObjectLoaderError(str(e)) from e

        config = None
        if ArchiveConfig and archpath:
            config = ArchiveConfig(archpath=archpath)

        try:
            reader = (
                self.client.bucket(bck_name=bck_name, provider=provider)
                .object(obj_name)
                .get_reader(archive_config=config)
            )
            return reader.read_all()
        except Exception as e:
            raise AISObjectLoaderError(
                f"Failed to fetch object '{obj_name}' from "
                f"{provider}://{bck_name}: {e}"
            ) from e

    def _fetch_sequential(self, fetch_tasks: List[Tuple[Any, str]]) -> None:
        """Fetch objects sequentially, fail-fast on error."""
        for manifest, url in fetch_tasks:
            content = self._fetch_object(url)
            inject_data_into_manifest(manifest, content)

    def _fetch_concurrent(self, fetch_tasks: List[Tuple[Any, str]]) -> None:
        """Fetch objects concurrently using a thread pool."""
        errors: List[Exception] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_manifest = {
                executor.submit(self._fetch_object, url): manifest
                for manifest, url in fetch_tasks
            }

            for future in as_completed(future_to_manifest):
                manifest = future_to_manifest[future]
                try:
                    content = future.result()
                    inject_data_into_manifest(manifest, content)
                except Exception as e:
                    errors.append(e)

        if errors:
            msg = f"{len(errors)} object(s) failed to fetch:\n"
            msg += "\n".join(f"  - {e}" for e in errors)
            raise AISObjectLoaderError(msg) from errors[0]
