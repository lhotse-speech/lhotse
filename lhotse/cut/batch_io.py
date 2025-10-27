from typing import Optional

from aistore.sdk import BatchRequest

from lhotse.cut import CutSet
from lhotse.features.base import Features
from lhotse.image import Image
from lhotse.audio.recording import Recording
from lhotse.array import Array, TemporalArray
from lhotse.serialization import get_aistore_client
from lhotse.utils import is_valid_url


FILE_TO_MEMORY_TYPE = {
    "numpy_files": "memory_raw",
    "lilcom_files": "memory_lilcom",
    "pillow_files": "memory_pillow",
}


class BatchMemoryLoaderError(Exception):
    """Base exception for BatchMemoryLoader operations."""


class BatchMemoryLoader:
    """
    :class:`BatchMemoryLoader` loads the entire data referenced by a :class:`CutSet`
    into memory with a single API call to the underlying storage provider.
    This operation is more efficient than sending separate requests for each item.

    :class:`BatchMemoryLoader` iterates over all data fields in a :class:`CutSet`, such
    as :class:`Recording`, :class:`Array`, :class:`Features`, etc. in order to append them
    to the request.

    Recording should have AudioSources of type "url".
    Features, Array, and TemporalArray should have storage_type "numpy_files" or "lilcom_files".
    Image should have storage_type "pillow_files".

    Currently, this capability is only supported with AIStore object store.

    Args:
        fault_tolerant: If True, skip cuts with missing data instead of raising errors
    """

    def __init__(self, fault_tolerant: bool = False) -> None:
        # Note: We don't have to, and shouldn't, accept aistore.Client() into the constructor.
        #       In Lhotse we're using the get_aistore_client() singleton instead to fetch it
        #       whenever we need it.
        client, _ = get_aistore_client()
        self._batch_loader = client.batch_loader()

        # When set to True drop the cuts for which at least one data field has failed.
        self._fault_tolerant = fault_tolerant

    def __call__(self, cuts: CutSet) -> CutSet:
        """
        Returns a new :class:`CutSet` with all data items loaded into memory.
        Internally uses ``cut.iter_data()`` to iterate all available fields.

        Example:
            >>> loader = BatchMemoryLoader(fault_tolerant=True)
            >>> memory_cuts = loader(cuts)

        Args:
            cuts (CutSet): CutSet to load into memory from AIStore

        Returns:
            CutSet: CutSet with all data items loaded into memory
        """

        if not isinstance(cuts, CutSet):
            raise BatchMemoryLoaderError("Input must be a CutSet instance")

        if len(cuts) == 0:
            return cuts

        if cuts.is_lazy:
            raise BatchMemoryLoaderError(
                "Lazy CutSets cannot be used with BatchMemoryLoader; "
                "make sure your CutSet represents a single batch of data and call .to_eager()."
            )

        # Gather the list of URLs for preparation of the request.
        urls = []
        for cut in cuts:
            for attr_name, manifest in cut.iter_data():
                if isinstance(manifest, Recording):
                    for source in manifest.sources:

                        # Source type must be URL for AIStore
                        if source.type != "url":
                            # raise BatchMemoryLoaderError(
                            #     f"Unsupported source type: '{source.type}'"
                            # )
                            continue

                        urls.append(source.source)

                elif isinstance(manifest, (Array, TemporalArray, Features, Image)):

                    # Storage type must be supported
                    if manifest.storage_type not in (
                        "numpy_files",
                        "lilcom_files",
                        "pillow_files",
                    ):
                        raise BatchMemoryLoaderError(
                            f"Unsupported storage type: '{manifest.storage_type}'"
                        )

                    path = f"{manifest.storage_path}/{manifest.storage_key}"

                    # Check if path is a url or filesystem path
                    if is_valid_url(path):
                        urls.append(path)

        # Construct batch request, send, receive objects
        batch_req = BatchRequest(
            continue_on_err=self._fault_tolerant, only_obj_name=False, streaming=True
        )

        for url in urls:
            provider, path = url.split("://")

            # Parse each URL to extract bucket name
            # We are using only_obj_name=False, so must have bucket
            bucket, obj_path = path.split("/", 1)

            # Parse URL to get valid output format
            output_fmt = self.get_output_format(url)

            # Get bucket from client
            client, _ = get_aistore_client()
            bck = client.bucket(bucket, provider)

            # If output format is none, then no archive
            if output_fmt is None:

                # Object name is second part
                obj = bck.object(obj_path)

                batch_req.add_object_request(obj=obj)

            else:
                # Split into objname, archpath
                obj_name, archpath = obj_path.split(output_fmt + "/", 1)

                # Obj name includes archive extension
                obj_name += output_fmt

                # Get object
                obj = bck.object(obj_name)

                batch_req.add_object_request(obj=obj, archpath=archpath)

                # Update output format type
                batch_req.output_format = output_fmt

        # Get stream of file data from AIStore
        received_blobs = self._batch_loader.get_batch(batch_req)

        for cut in cuts:
            # Note: the modifications done below are in-place
            for attr_name, manifest in cut.iter_data():
                batch_resp, blob = next(received_blobs)

                # Check if the returned data is missing
                if not self._fault_tolerant and batch_resp.is_missing:
                    raise BatchMemoryLoaderError(
                        f"Missing data, could not get {attr_name} for cut {cut.id}"
                    )

                if isinstance(manifest, Recording):
                    for source in manifest.sources:
                        source.type = "memory"
                        source.source = blob
                elif isinstance(manifest, (Array, TemporalArray, Features, Image)):
                    manifest.storage_type = FILE_TO_MEMORY_TYPE[manifest.storage_type]
                    manifest.storage_path = ""
                    manifest.storage_key = blob

        return cuts

    def get_output_format(self, path: str) -> Optional[str]:
        """
        Returns the output format type from a provided path if it contains
        a supported archival output type.

        Returns:
            str (optional): Output type (file extension) if path contains 
            supported archive type, else None
        """
        for ext in (".tar.gz", ".tar", ".tgz"):
            if ext in path:
                return ext

        return None
