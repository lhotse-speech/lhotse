from lhotse import CutSet, Features, Image, Recording
from lhotse.array import Array, TemporalArray


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
    """

    def __init__(self) -> None:
        # TODO(pzelasko): introduce necessary config args, or remove __init__
        # Note: We don't have to, and shouldn't, accept aistore.Client() into the constructor.
        #       In Lhotse we're using the get_aistore_client() singleton instead to fetch it
        #       whenever we need it.
        pass

    def __call__(self, cuts: CutSet) -> CutSet:
        """
        Returns a new :class:`CutSet` with all data items loaded into memory.
        Internally uses ``cut.iter_data()`` to iterate all available fields
        """
        assert not cuts.is_lazy, (
            "Lazy CutSets cannot be used with BatchMemoryLoader; "
            "make sure your CutSet represents a single batch of data and call .to_eager()."
        )

        # Gather the list of URLs for preparation of the request.
        urls = []
        for cut in cuts:
            for attr_name, manifest in cut.iter_data():
                if isinstance(manifest, Recording):
                    for source in manifest.sources:
                        assert (
                            source.type == "url"
                        ), f"Unsupported source type: '{source.type}'"
                        urls.append(source.source)
                elif isinstance(manifest, (Array, TemporalArray, Features, Image)):
                    assert manifest.storage_type in (
                        "numpy_files",
                        "lilcom_files",
                        "pillow_files",
                    ), f"Unsupported storage type: '{manifest.storage_type}'"
                    urls.append(f"{manifest.storage_path}/{manifest.storage_key}")

        # TODO(pzelasko): construct batch request, send, receive objects
        request = ...
        received_blobs = [...]

        # Populate the binary data into manifests.
        # TODO: might need to introduce error handling here - perhaps as a configurable constructor option
        #       ``fault_tolerant: bool = False`; by default, raise an error,
        #       when set to True drop the cuts for which at least one data field has failed.
        received_blobs = iter(received_blobs)
        for cut in cuts:
            # Note: the modifications done below are in-place
            for attr_name, manifest in cut.iter_data():
                blob: bytes = next(received_blobs)
                if isinstance(manifest, Recording):
                    for source in manifest.sources:
                        source.type = "memory"
                        source.source = blob
                elif isinstance(manifest, (Array, TemporalArray, Features, Image)):
                    manifest.storage_type = FILE_TO_MEMORY_TYPE[manifest.storage_type]
                    manifest.storage_path = ""
                    manifest.storage_key = blob

        return cuts


FILE_TO_MEMORY_TYPE = {
    "numpy_files": "memory_raw",
    "lilcom_files": "memory_lilcom",
    "pillow_files": "memory_pillow",
}
