import pickle
import threading
from abc import ABCMeta, abstractmethod
from functools import lru_cache
from math import ceil, floor
from pathlib import Path
from typing import List, Optional, Type, Union

import lilcom
import numpy as np

from lhotse.array import Array, TemporalArray
from lhotse.caching import dynamic_lru_cache
from lhotse.utils import Pathlike, Seconds, SmartOpen, is_module_available


class FeaturesWriter(metaclass=ABCMeta):
    """
    ``FeaturesWriter`` defines the interface of how to store numpy arrays in a particular storage backend.
    This backend could either be:

    - separate files on a local filesystem;
    - a single file with multiple arrays;
    - cloud storage;
    - etc.

    Each class inheriting from ``FeaturesWriter`` must define:

    - the ``write()`` method, which defines the storing operation
        (accepts a ``key`` used to place the ``value`` array in the storage);
    - the ``storage_path()`` property, which is either a common directory for the files,
        the name of the file storing multiple arrays, name of the cloud bucket, etc.
    - the ``name()`` property that is unique to this particular storage mechanism -
        it is stored in the features manifests (metadata) and used to automatically deduce
        the backend when loading the features.

    Each :class:`.FeaturesWriter` can also be used as a context manager, as some implementations
    might need to free a resource after the writing is finalized. By default nothing happens
    in the context manager functions, and this can be modified by the inheriting subclasses.

    Example::

        >>> with MyWriter('some/path') as storage:
        ...     extractor.extract_from_recording_and_store(recording, storage)

    The features loading must be defined separately in a class inheriting from :class:`FeaturesReader`.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def storage_path(self) -> str:
        ...

    @abstractmethod
    def write(self, key: str, value: np.ndarray) -> str:
        ...

    def store_array(
        self,
        key: str,
        value: np.ndarray,
        frame_shift: Optional[Seconds] = None,
        temporal_dim: Optional[int] = None,
        start: Seconds = 0,
    ) -> Union[Array, TemporalArray]:
        """
        Store a numpy array in the underlying storage and return a manifest
        describing how to retrieve the data.

        If the array contains a temporal dimension (e.g. it represents the
        frame-level features, alignment, posteriors, etc. of an utterance)
        then ``temporal_dim`` and ``frame_shift`` may be specified to enable
        downstream padding, truncating, and partial reads of the array.

        :param key: An ID that uniquely identifies the array.
        :param value: The array to be stored.
        :param frame_shift: Optional float, when the array has a temporal dimension
            it indicates how much time has passed between the starts of consecutive frames
            (expressed in seconds).
        :param temporal_dim: Optional int, when the array has a temporal dimension,
            it indicates which dim to interpret as temporal.
        :param start: Float, when the array is temporal, it indicates what is the offset
            of the array w.r.t. the start of recording. Useful for reading subsets
            of an array when it represents something computed from long recordings.
            Ignored for non-temporal arrays.
        :return: A manifest of type :class:`~lhotse.array.Array` or
            :class:`~lhotse.array.TemporalArray`, depending on the input arguments.
        """
        is_temporal = frame_shift is not None and temporal_dim is not None
        if not is_temporal:
            assert all(arg is None for arg in [frame_shift, temporal_dim]), (
                "frame_shift and temporal_dim have to be both None or both set "
                f"(got frame_shift={frame_shift}, temporal_dim={temporal_dim})."
            )

        storage_key = self.write(key, value)
        array = Array(
            storage_type=self.name,
            storage_path=self.storage_path,
            storage_key=storage_key,
            shape=list(value.shape),
        )

        if not is_temporal:
            return array

        return TemporalArray(
            array=array,
            temporal_dim=temporal_dim,
            frame_shift=frame_shift,
            start=start,
        )

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        ...


class FeaturesReader(metaclass=ABCMeta):
    """
    ``FeaturesReader`` defines the interface of how to load numpy arrays from a particular storage backend.
    This backend could either be:

    - separate files on a local filesystem;
    - a single file with multiple arrays;
    - cloud storage;
    - etc.

    Each class inheriting from ``FeaturesReader`` must define:

    - the ``read()`` method, which defines the loading operation
        (accepts the ``key`` to locate the array in the storage and return it).
        The read method should support selecting only a subset of the feature matrix,
        with the bounds expressed as arguments ``left_offset_frames`` and ``right_offset_frames``.
        It's up to the Reader implementation to load only the required part or trim it to that
        range only after loading. It is assumed that the time dimension is always the first one.
    - the ``name()`` property that is unique to this particular storage mechanism -
        it is stored in the features manifests (metadata) and used to automatically deduce
        the backend when loading the features.

    The features writing must be defined separately in a class inheriting from ``FeaturesWriter``.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def read(
        self,
        key: str,
        left_offset_frames: int = 0,
        right_offset_frames: Optional[int] = None,
    ) -> np.ndarray:
        ...


READER_BACKENDS = {}
WRITER_BACKENDS = {}


def available_storage_backends() -> List[str]:
    return sorted(set(READER_BACKENDS).intersection(WRITER_BACKENDS))


def register_reader(cls):
    """
    Decorator used to add a new ``FeaturesReader`` to Lhotse's registry.

    Example::

        @register_reader
        class MyFeatureReader(FeatureReader):
            ...
    """
    READER_BACKENDS[cls.name] = cls
    return cls


def register_writer(cls):
    """
    Decorator used to add a new ``FeaturesWriter`` to Lhotse's registry.

    Example::

        @register_writer
        class MyFeatureWriter(FeatureWriter):
            ...
    """
    WRITER_BACKENDS[cls.name] = cls
    return cls


def get_reader(name: str) -> Type[FeaturesReader]:
    """
    Find a ``FeaturesReader`` sub-class that corresponds to the provided ``name`` and return its type.

    Example:

        reader_type = get_reader("lilcom_files")
        reader = reader_type("/storage/features/")
    """
    return READER_BACKENDS.get(name)


def get_writer(name: str) -> Type[FeaturesWriter]:
    """
    Find a ``FeaturesWriter`` sub-class that corresponds to the provided ``name`` and return its type.

    Example:

        writer_type = get_writer("lilcom_files")
        writer = writer_type("/storage/features/")
    """
    return WRITER_BACKENDS.get(name)


"""
Lilcom-compressed numpy arrays, stored in separate files on the filesystem.
"""


@register_reader
class LilcomFilesReader(FeaturesReader):
    """
    Reads Lilcom-compressed files from a directory on the local filesystem.
    ``storage_path`` corresponds to the directory path;
    ``storage_key`` for each utterance is the name of the file in that directory.
    """

    name = "lilcom_files"

    def __init__(self, storage_path: Pathlike, *args, **kwargs):
        super().__init__()
        self.storage_path = Path(storage_path)

    @dynamic_lru_cache
    def read(
        self,
        key: str,
        left_offset_frames: int = 0,
        right_offset_frames: Optional[int] = None,
    ) -> np.ndarray:
        with open(self.storage_path / key, "rb") as f:
            arr = lilcom.decompress(f.read())
        return arr[left_offset_frames:right_offset_frames]


@register_writer
class LilcomFilesWriter(FeaturesWriter):
    """
    Writes Lilcom-compressed files to a directory on the local filesystem.
    ``storage_path`` corresponds to the directory path;
    ``storage_key`` for each utterance is the name of the file in that directory.
    """

    name = "lilcom_files"

    def __init__(self, storage_path: Pathlike, tick_power: int = -5, *args, **kwargs):
        super().__init__()
        self.storage_path_ = Path(storage_path)
        self.storage_path_.mkdir(parents=True, exist_ok=True)
        self.tick_power = tick_power

    @property
    def storage_path(self) -> str:
        return str(self.storage_path_)

    def write(self, key: str, value: np.ndarray) -> str:
        # Introduce a sub-directory that starts with the first 3 characters of the key, that is typically
        # an auto-generated hash. This allows to avoid filesystem performance problems related to storing
        # too many files in a single directory.
        subdir = self.storage_path_ / key[:3]
        subdir.mkdir(exist_ok=True)
        output_features_path = (subdir / key).with_suffix(".llc")
        serialized_feats = lilcom.compress(value, tick_power=self.tick_power)
        with open(output_features_path, "wb") as f:
            f.write(serialized_feats)
        # Include sub-directory in the key, e.g. "abc/abcdef.llc"
        return "/".join(output_features_path.parts[-2:])


"""
Non-compressed numpy arrays, stored in separate files on the filesystem.
"""


@register_reader
class NumpyFilesReader(FeaturesReader):
    """
    Reads non-compressed numpy arrays from files in a directory on the local filesystem.
    ``storage_path`` corresponds to the directory path;
    ``storage_key`` for each utterance is the name of the file in that directory.
    """

    name = "numpy_files"

    def __init__(self, storage_path: Pathlike, *args, **kwargs):
        super().__init__()
        self.storage_path = Path(storage_path)

    @dynamic_lru_cache
    def read(
        self,
        key: str,
        left_offset_frames: int = 0,
        right_offset_frames: Optional[int] = None,
    ) -> np.ndarray:
        arr = np.load(self.storage_path / key, allow_pickle=False)
        return arr[left_offset_frames:right_offset_frames]


@register_writer
class NumpyFilesWriter(FeaturesWriter):
    """
    Writes non-compressed numpy arrays to files in a directory on the local filesystem.
    ``storage_path`` corresponds to the directory path;
    ``storage_key`` for each utterance is the name of the file in that directory.
    """

    name = "numpy_files"

    def __init__(self, storage_path: Pathlike, *args, **kwargs):
        super().__init__()
        self.storage_path_ = Path(storage_path)
        self.storage_path_.mkdir(parents=True, exist_ok=True)

    @property
    def storage_path(self) -> str:
        return str(self.storage_path_)

    def write(self, key: str, value: np.ndarray) -> str:
        # Introduce a sub-directory that starts with the first 3 characters of the key, that is typically
        # an auto-generated hash. This allows to avoid filesystem performance problems related to storing
        # too many files in a single directory.
        subdir = self.storage_path_ / key[:3]
        subdir.mkdir(exist_ok=True)
        output_features_path = (subdir / key).with_suffix(".npy")
        np.save(output_features_path, value, allow_pickle=False)
        # Include sub-directory in the key, e.g. "abc/abcdef.npy"
        return "/".join(output_features_path.parts[-2:])


"""
Non-compressed numpy arrays, stored in HDF5 file.
"""


@lru_cache(maxsize=None)
def lookup_cache_or_open(storage_path: str):
    """
    Helper internal function used in HDF5 readers.
    It opens the HDF files and keeps their handles open in a global program cache
    to avoid excessive amount of syscalls when the Reader class is instantiated
    and destroyed in a loop repeatedly (frequent use-case).

    The file handles can be freed at any time by calling ``close_cached_file_handles()``.
    """
    import h5py

    return h5py.File(storage_path, "r")


@lru_cache(maxsize=None)
def lookup_cache_or_open_regular_file(storage_path: str):
    """
    Helper internal function used in "fast" file readers.
    It opens regular files and keeps their handles open in a global program cache
    to avoid excessive amount of syscalls when the Reader class is instantiated
    and destroyed in a loop repeatedly (frequent use-case).

    The file handles can be freed at any time by calling ``close_cached_file_handles()``.
    """
    f = open(storage_path, "rb")
    return f


@lru_cache(maxsize=None)
def lookup_chunk_size(h5_file_handle) -> int:
    """
    Helper internal function to retrieve the chunk size from an HDF5 file.
    Helps avoid unnecessary repeated disk reads.
    """
    return h5_file_handle[CHUNK_SIZE_KEY][()]  # [()] retrieves a scalar


def close_cached_file_handles() -> None:
    """Closes the cached file handles in ``lookup_cache_or_open`` (see its docs for more details)."""
    lookup_cache_or_open_regular_file.cache_clear()
    lookup_cache_or_open.cache_clear()
    lookup_chunk_size.cache_clear()


@register_reader
class NumpyHdf5Reader(FeaturesReader):
    """
    Reads non-compressed numpy arrays from a HDF5 file with a "flat" layout.
    Each array is stored as a separate HDF ``Dataset`` because their shapes (numbers of frames) may vary.
    ``storage_path`` corresponds to the HDF5 file path;
    ``storage_key`` for each utterance is the key corresponding to the array (i.e. HDF5 "Group" name).
    """

    name = "numpy_hdf5"

    def __init__(self, storage_path: Pathlike, *args, **kwargs):
        super().__init__()
        self.hdf = lookup_cache_or_open(storage_path)

    @dynamic_lru_cache
    def read(
        self,
        key: str,
        left_offset_frames: int = 0,
        right_offset_frames: Optional[int] = None,
    ) -> np.ndarray:
        # (pzelasko): If I understand HDF5/h5py correctly, this implementation reads only
        # the requested slice of the array into memory - but don't take my word for it.
        return self.hdf[key][left_offset_frames:right_offset_frames]


@register_writer
class NumpyHdf5Writer(FeaturesWriter):
    """
    Writes non-compressed numpy arrays to a HDF5 file with a "flat" layout.
    Each array is stored as a separate HDF ``Dataset`` because their shapes (numbers of frames) may vary.
    ``storage_path`` corresponds to the HDF5 file path;
    ``storage_key`` for each utterance is the key corresponding to the array (i.e. HDF5 "Group" name).

    Internally, this class opens the file lazily so that this object can be passed between processes
    without issues. This simplifies the parallel feature extraction code.
    """

    name = "numpy_hdf5"

    def __init__(self, storage_path: Pathlike, mode: str = "w", *args, **kwargs):
        """
        :param storage_path: Path under which we'll create the HDF5 file.
            We will add a ``.h5`` suffix if it is not already in ``storage_path``.
        :param mode: Modes supported by h5py:
            w        Create file, truncate if exists (default)
            w- or x  Create file, fail if exists
            a        Read/write if exists, create otherwise
        """
        super().__init__()
        import h5py

        self.storage_path_ = Path(storage_path).with_suffix(".h5")
        self.hdf = h5py.File(self.storage_path, mode=mode)

    @property
    def storage_path(self) -> str:
        return str(self.storage_path_)

    def write(self, key: str, value: np.ndarray) -> str:
        self.hdf.create_dataset(key, data=value)
        return key

    def close(self) -> None:
        return self.hdf.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


"""
Lilcom-compressed numpy arrays, stored in HDF5 file.
"""


@register_reader
class LilcomHdf5Reader(FeaturesReader):
    """
    Reads lilcom-compressed numpy arrays from a HDF5 file with a "flat" layout.
    Each array is stored as a separate HDF ``Dataset`` because their shapes (numbers of frames) may vary.
    ``storage_path`` corresponds to the HDF5 file path;
    ``storage_key`` for each utterance is the key corresponding to the array (i.e. HDF5 "Group" name).
    """

    name = "lilcom_hdf5"

    def __init__(self, storage_path: Pathlike, *args, **kwargs):
        super().__init__()
        self.hdf = lookup_cache_or_open(storage_path)

    @dynamic_lru_cache
    def read(
        self,
        key: str,
        left_offset_frames: int = 0,
        right_offset_frames: Optional[int] = None,
    ) -> np.ndarray:
        # This weird indexing with [()] is a replacement for ".value" attribute,
        # that got deprecated with the following warning:
        # H5pyDeprecationWarning: dataset.value has been deprecated. Use dataset[()] instead.
        #     arr = lilcom.decompress(self.hdf[key].value.tobytes())
        arr = lilcom.decompress(self.hdf[key][()].tobytes())
        return arr[left_offset_frames:right_offset_frames]


@register_writer
class LilcomHdf5Writer(FeaturesWriter):
    """
    Writes lilcom-compressed numpy arrays to a HDF5 file with a "flat" layout.
    Each array is stored as a separate HDF ``Dataset`` because their shapes (numbers of frames) may vary.
    ``storage_path`` corresponds to the HDF5 file path;
    ``storage_key`` for each utterance is the key corresponding to the array (i.e. HDF5 "Group" name).
    """

    name = "lilcom_hdf5"

    def __init__(
        self,
        storage_path: Pathlike,
        tick_power: int = -5,
        mode: str = "w",
        *args,
        **kwargs,
    ):
        """
        :param storage_path: Path under which we'll create the HDF5 file.
            We will add a ``.h5`` suffix if it is not already in ``storage_path``.
        :param tick_power: Determines the lilcom compression accuracy;
            the input will be compressed to integer multiples of 2^tick_power.
        :param mode: Modes supported by h5py:
            w        Create file, truncate if exists (default)
            w- or x  Create file, fail if exists
            a        Read/write if exists, create otherwise
        """
        super().__init__()
        import h5py

        self.storage_path_ = Path(storage_path).with_suffix(".h5")
        self.hdf = h5py.File(self.storage_path, mode=mode)
        self.tick_power = tick_power

    @property
    def storage_path(self) -> str:
        return str(self.storage_path_)

    def write(self, key: str, value: np.ndarray) -> str:
        serialized_feats = lilcom.compress(value, tick_power=self.tick_power)
        self.hdf.create_dataset(key, data=np.void(serialized_feats))
        return key

    def close(self) -> None:
        return self.hdf.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


"""
Lilcom-compressed numpy arrays, stored in HDF5 file, that store chunked features.
They are suitable for storing features for long recordings since they are able to
retrieve small chunks instead of full matrices.
"""

CHUNK_SIZE_KEY = "__LHOTSE_INTERNAL_CHUNK_SIZE__"


@register_reader
class ChunkedLilcomHdf5Reader(FeaturesReader):
    """
    Reads lilcom-compressed numpy arrays from a HDF5 file with chunked lilcom storage.
    Each feature matrix is stored in an array of chunks - binary data compressed with lilcom.
    Upon reading, we check how many chunks need to be retrieved to avoid excessive I/O.

    ``storage_path`` corresponds to the HDF5 file path;
    ``storage_key`` for each utterance is the key corresponding to the array (i.e. HDF5 "Group" name).
    """

    name = "chunked_lilcom_hdf5"

    def __init__(self, storage_path: Pathlike, *args, **kwargs):
        super().__init__()
        self.hdf = lookup_cache_or_open(storage_path)

    @dynamic_lru_cache
    def read(
        self,
        key: str,
        left_offset_frames: int = 0,
        right_offset_frames: Optional[int] = None,
    ) -> np.ndarray:
        # First, determine which range of chunks need to be read.
        chunk_size = lookup_chunk_size(self.hdf)
        left_chunk_idx = floor(left_offset_frames / chunk_size)
        if right_offset_frames is not None:
            right_chunk_idx = ceil(right_offset_frames / chunk_size)
        else:
            right_chunk_idx = None

        # Read, decode, concat
        decompressed_chunks = [
            lilcom.decompress(data.tobytes())
            for data in self.hdf[key][left_chunk_idx:right_chunk_idx]
        ]
        if decompressed_chunks:
            arr = np.concatenate(decompressed_chunks, axis=0)
        else:
            arr = np.array([])

        # Determine what piece of decoded data should be returned;
        # we offset the input offsets by left_chunk_idx * chunk_size.
        shift_frames = chunk_size * left_chunk_idx
        left_offset_shift = left_offset_frames - shift_frames
        if right_offset_frames is not None:
            right_offset_shift = right_offset_frames - shift_frames
        else:
            right_offset_shift = None

        return arr[left_offset_shift:right_offset_shift]


@register_writer
class ChunkedLilcomHdf5Writer(FeaturesWriter):
    """
    Writes lilcom-compressed numpy arrays to a HDF5 file with chunked lilcom storage.
    Each feature matrix is stored in an array of chunks - binary data compressed with lilcom.
    Upon reading, we check how many chunks need to be retrieved to avoid excessive I/O.

    ``storage_path`` corresponds to the HDF5 file path;
    ``storage_key`` for each utterance is the key corresponding to the array (i.e. HDF5 "Group" name).
    """

    name = "chunked_lilcom_hdf5"

    def __init__(
        self,
        storage_path: Pathlike,
        tick_power: int = -5,
        chunk_size: int = 100,
        mode: str = "w",
        *args,
        **kwargs,
    ):
        """
        :param storage_path: Path under which we'll create the HDF5 file.
            We will add a ``.h5`` suffix if it is not already in ``storage_path``.
        :param tick_power: Determines the lilcom compression accuracy;
            the input will be compressed to integer multiples of 2^tick_power.
        :param chunk_size: How many frames to store per chunk.
            Too low a number will require many reads for long feature matrices,
            too high a number will require to read more redundant data.
        :param mode: Modes supported by h5py:
            w        Create file, truncate if exists (default)
            w- or x  Create file, fail if exists
            a        Read/write if exists, create otherwise
        """
        super().__init__()
        import h5py

        self.storage_path_ = Path(storage_path).with_suffix(".h5")
        self.tick_power = tick_power
        self.chunk_size = chunk_size
        self.hdf = h5py.File(self.storage_path, mode=mode)
        if CHUNK_SIZE_KEY in self.hdf:
            retrieved_chunk_size = self.hdf[CHUNK_SIZE_KEY][()]
            assert retrieved_chunk_size == CHUNK_SIZE_KEY, (
                f"Error: attempted to write with chunk size {self.chunk_size} to an h5py file that "
                f"was created with chunk size {retrieved_chunk_size}."
            )
        else:
            self.hdf.create_dataset(CHUNK_SIZE_KEY, data=self.chunk_size)

    @property
    def storage_path(self) -> str:
        return str(self.storage_path_)

    def write(self, key: str, value: np.ndarray) -> str:
        import h5py
        from lhotse.features.compression import lilcom_compress_chunked

        serialized_feats = lilcom_compress_chunked(
            value, tick_power=self.tick_power, chunk_size=self.chunk_size
        )
        dset = self.hdf.create_dataset(
            key,
            dtype=h5py.vlen_dtype(np.dtype("uint8")),
            shape=(len(serialized_feats),),
        )
        for idx, feat in enumerate(serialized_feats):
            dset[idx] = np.frombuffer(feat, dtype=np.uint8)
        return key

    def close(self) -> None:
        return self.hdf.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


"""
Lilcom-compressed numpy arrays, stored in my own custom format file, that store chunked features.
They are suitable for storing features for long recordings since they are able to
retrieve small chunks instead of full matrices.
"""

CHUNKY_FORMAT_CHUNK_SIZE = 500  # constant


@register_reader
class LilcomChunkyReader(FeaturesReader):
    """
    Reads lilcom-compressed numpy arrays from a binary file with chunked lilcom storage.
    Each feature matrix is stored in an array of chunks - binary data compressed with lilcom.
    Upon reading, we check how many chunks need to be retrieved to avoid excessive I/O.

    ``storage_path`` corresponds to the binary file path.

    ``storage_key`` for each utterance is a comma separated list of offsets in the file.
    The first number is the offset for the whole array,
    and the following numbers are relative offsets for each chunk.
    These offsets are relative to the previous chunk start.
    """

    name = "lilcom_chunky"
    CHUNK_SIZE = CHUNKY_FORMAT_CHUNK_SIZE

    def __init__(self, storage_path: Pathlike, *args, **kwargs):
        super().__init__()
        self.file = lookup_cache_or_open_regular_file(storage_path)
        self.lock = threading.Lock()

    @dynamic_lru_cache
    def read(
        self,
        key: str,
        left_offset_frames: int = 0,
        right_offset_frames: Optional[int] = None,
    ) -> np.ndarray:
        # First, determine which range of chunks need to be read.
        left_chunk_idx = floor(left_offset_frames / self.CHUNK_SIZE)
        if right_offset_frames is not None:
            # Note: +1 is to include the end of the last chunk
            right_chunk_idx = ceil(right_offset_frames / self.CHUNK_SIZE) + 1
        else:
            right_chunk_idx = None

        chunk_offsets = list(map(int, key.split(",")))
        chunk_offsets = np.cumsum(chunk_offsets)
        chunk_offsets = chunk_offsets[left_chunk_idx:right_chunk_idx]

        chunk_data = []
        for offset, end in pairwise(chunk_offsets):
            # We need to use locks to avoid race conditions between seek
            # and read in multi-threaded reads.
            with self.lock:
                self.file.seek(offset)
                chunk_data.append(self.file.read(end - offset))

        # Read, decode, concat
        decompressed_chunks = [lilcom.decompress(data) for data in chunk_data]
        if decompressed_chunks:
            arr = np.concatenate(decompressed_chunks, axis=0)
        else:
            arr = np.array([])

        # Determine what piece of decoded data should be returned;
        # we offset the input offsets by left_chunk_idx * chunk_size.
        shift_frames = self.CHUNK_SIZE * left_chunk_idx
        left_offset_shift = left_offset_frames - shift_frames
        if right_offset_frames is not None:
            right_offset_shift = right_offset_frames - shift_frames
        else:
            right_offset_shift = None

        return arr[left_offset_shift:right_offset_shift]


@register_writer
class LilcomChunkyWriter(FeaturesWriter):
    """
    Writes lilcom-compressed numpy arrays to a binary file with chunked lilcom storage.
    Each feature matrix is stored in an array of chunks - binary data compressed with lilcom.
    Upon reading, we check how many chunks need to be retrieved to avoid excessive I/O.

    ``storage_path`` corresponds to the binary file path.

    ``storage_key`` for each utterance is a comma separated list of offsets in the file.
    The first number is the offset for the whole array,
    and the following numbers are relative offsets for each chunk.
    These offsets are relative to the previous chunk start.
    """

    name = "lilcom_chunky"
    CHUNK_SIZE = CHUNKY_FORMAT_CHUNK_SIZE

    def __init__(
        self,
        storage_path: Pathlike,
        tick_power: int = -5,
        mode: str = "wb",
        *args,
        **kwargs,
    ):
        """
        :param storage_path: Path under which we'll create the binary file.
        :param tick_power: Determines the lilcom compression accuracy;
            the input will be compressed to integer multiples of 2^tick_power.
        :param chunk_size: How many frames to store per chunk.
            Too low a number will require many reads for long feature matrices,
            too high a number will require to read more redundant data.
        :param mode: Modes, one of: "w" (write) or "a" (append); can be "wb" and "ab", "b" is implicit
        """
        super().__init__()

        if "b" not in mode:
            mode = mode + "b"
        assert mode == "wb" or "ab"

        # ".lca" -> "lilcom chunky archive"
        self.storage_path_ = Path(storage_path).with_suffix(".lca")
        self.tick_power = tick_power
        self.file = open(self.storage_path, mode=mode)
        self.curr_offset = self.file.tell()

    @property
    def storage_path(self) -> str:
        return str(self.storage_path_)

    def write(self, key: str, value: np.ndarray) -> str:
        from lhotse.features.compression import lilcom_compress_chunked

        serialized_feats = lilcom_compress_chunked(
            value, tick_power=self.tick_power, chunk_size=self.CHUNK_SIZE
        )
        offsets = [self.curr_offset]
        for idx, feat in enumerate(serialized_feats):
            nbytes = self.file.write(feat)
            offsets.append(nbytes)
            self.curr_offset += nbytes

        # Returns keys like: "14601,31,23,42".
        # The first number is the offset for the whole array,
        # and the following numbers are relative offsets for each chunk.
        # These offsets are relative to the previous chunk start.
        return ",".join(map(str, offsets))

    def close(self) -> None:
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


"""
Lilcom-compressed URL writers
"""


@register_reader
class LilcomURLReader(FeaturesReader):
    """
    Downloads Lilcom-compressed files from a URL (S3, GCP, Azure, HTTP, etc.).
    ``storage_path`` corresponds to the root URL (e.g. "s3://my-data-bucket")
    ``storage_key`` will be concatenated to ``storage_path`` to form a full URL (e.g. "my-feature-file.llc")

    .. caution::
        Requires ``smart_open`` to be installed (``pip install smart_open``).
    """

    name = "lilcom_url"

    def __init__(self, storage_path: Pathlike, *args, **kwargs):
        super().__init__()
        self.base_url = str(storage_path)
        # We are manually adding the slash to join the base URL and the key.
        if self.base_url.endswith("/"):
            self.base_url = self.base_url[:-1]

    @dynamic_lru_cache
    def read(
        self,
        key: str,
        left_offset_frames: int = 0,
        right_offset_frames: Optional[int] = None,
    ) -> np.ndarray:
        # We are manually adding the slash to join the base URL and the key.
        if key.startswith("/"):
            key = key[1:]
        with SmartOpen.open(f"{self.base_url}/{key}", "rb") as f:
            arr = lilcom.decompress(f.read())
        return arr[left_offset_frames:right_offset_frames]


@register_writer
class LilcomURLWriter(FeaturesWriter):
    """
    Writes Lilcom-compressed files to a URL (S3, GCP, Azure, HTTP, etc.).
    ``storage_path`` corresponds to the root URL (e.g. "s3://my-data-bucket")
    ``storage_key`` will be concatenated to ``storage_path`` to form a full URL (e.g. "my-feature-file.llc")

    .. caution::
        Requires ``smart_open`` to be installed (``pip install smart_open``).
    """

    name = "lilcom_url"

    def __init__(self, storage_path: Pathlike, tick_power: int = -5, *args, **kwargs):
        super().__init__()
        self.base_url = str(storage_path)
        # We are manually adding the slash to join the base URL and the key.
        if self.base_url.endswith("/"):
            self.base_url = self.base_url[:-1]
        self.tick_power = tick_power

    @property
    def storage_path(self) -> str:
        return self.base_url

    def write(self, key: str, value: np.ndarray) -> str:
        # We are manually adding the slash to join the base URL and the key.
        if key.startswith("/"):
            key = key[1:]
        # Add lilcom extension.
        if not key.endswith(".llc"):
            key = key + ".llc"
        output_features_url = f"{self.base_url}/{key}"
        serialized_feats = lilcom.compress(value, tick_power=self.tick_power)
        with SmartOpen.open(output_features_url, "wb") as f:
            f.write(serialized_feats)
        return key


"""
Kaldi-compatible feature reader
"""


@register_reader
class KaldiReader(FeaturesReader):
    """
    Reads Kaldi's "feats.scp" file using kaldi_native_io.
    ``storage_path`` corresponds to the path to ``feats.scp``.
    ``storage_key`` corresponds to the utterance-id in Kaldi.

    .. caution::
        Requires ``kaldi_native_io`` to be installed (``pip install kaldi_native_io``).
    """

    name = "kaldiio"

    def __init__(self, storage_path: Pathlike, *args, **kwargs):
        if not is_module_available("kaldi_native_io"):
            raise ValueError(
                "To read Kaldi feats.scp, please 'pip install kaldi_native_io' first."
            )
        import kaldi_native_io

        super().__init__()
        self.storage_path = storage_path
        self.storage = kaldi_native_io.RandomAccessFloatMatrixReader(
            f"scp:{self.storage_path}"
        )

    @dynamic_lru_cache
    def read(
        self,
        key: str,
        left_offset_frames: int = 0,
        right_offset_frames: Optional[int] = None,
    ) -> np.ndarray:
        arr = np.copy(self.storage[key])
        return arr[left_offset_frames:right_offset_frames]


@register_writer
class KaldiWriter(FeaturesWriter):
    """
    Write data to Kaldi's "feats.scp" and "feats.ark" files using kaldi_native_io.
    ``storage_path`` corresponds to a directory where we'll create "feats.scp"
    and "feats.ark" files.
    ``storage_key`` corresponds to the utterance-id in Kaldi.

    The following ``compression_method`` values are supported by kaldi_native_io::

        kAutomaticMethod = 1
        kSpeechFeature = 2
        kTwoByteAuto = 3
        kTwoByteSignedInteger = 4
        kOneByteAuto = 5
        kOneByteUnsignedInteger = 6
        kOneByteZeroOne = 7

    .. note:: Setting compression_method works only with 2D arrays.

    Example::

        >>> data = np.random.randn(131, 80)
        >>> with KaldiWriter('featdir') as w:
        ...     w.write('utt1', data)
        >>> reader = KaldiReader('featdir/feats.scp')
        >>> read_data = reader.read('utt1')
        >>> np.testing.assert_equal(data, read_data)

    .. caution::
        Requires ``kaldi_native_io`` to be installed (``pip install kaldi_native_io``).
    """

    name = "kaldiio"

    def __init__(
        self,
        storage_path: Pathlike,
        compression_method: int = 1,
        *args,
        **kwargs,
    ):
        if not is_module_available("kaldi_native_io"):
            raise ValueError(
                "To read Kaldi feats.scp, please 'pip install kaldi_native_io' first."
            )
        import kaldi_native_io

        super().__init__()
        self.storage_dir = Path(storage_path)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.storage_path_ = str(self.storage_dir / "feats.scp")
        self.storage = kaldi_native_io.CompressedMatrixWriter(
            f"ark,scp:{self.storage_dir}/feats.ark,{self.storage_dir}/feats.scp"
        )
        self.compression_method = kaldi_native_io.CompressionMethod(compression_method)

    @property
    def storage_path(self) -> str:
        return self.storage_path_

    def write(self, key: str, value: np.ndarray) -> str:
        self.storage.write(key, value, self.compression_method)
        return key

    def close(self) -> None:
        return self.storage.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


"""
In-memory reader/writer
"""


def get_memory_writer(name: str):
    assert "memory" in name
    return get_writer(name)


@register_reader
class MemoryLilcomReader(FeaturesReader):
    """ """

    name = "memory_lilcom"

    def __init__(self, *args, **kwargs):
        pass

    @dynamic_lru_cache
    def read(
        self,
        raw_data: bytes,
        left_offset_frames: int = 0,
        right_offset_frames: Optional[int] = None,
    ) -> np.ndarray:
        arr = lilcom.decompress(raw_data)
        return arr[left_offset_frames:right_offset_frames]


@register_writer
class MemoryLilcomWriter(FeaturesWriter):
    """ """

    name = "memory_lilcom"

    def __init__(self, *args, **kwargs):
        pass

    @property
    def storage_path(self) -> None:
        return None

    def write(self, key: str, value: np.ndarray) -> bytes:
        return lilcom.compress(value)

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


@register_reader
class MemoryRawReader(FeaturesReader):
    """ """

    name = "memory_raw"

    def __init__(self, *args, **kwargs):
        pass

    @dynamic_lru_cache
    def read(
        self,
        raw_data: bytes,
        left_offset_frames: int = 0,
        right_offset_frames: Optional[int] = None,
    ) -> np.ndarray:
        arr = pickle.loads(raw_data)
        return arr[left_offset_frames:right_offset_frames]


@register_writer
class MemoryRawWriter(FeaturesWriter):
    """ """

    name = "memory_raw"

    def __init__(self, *args, **kwargs):
        pass

    @property
    def storage_path(self) -> None:
        return None

    def write(self, key: str, value: np.ndarray) -> bytes:
        return pickle.dumps(value)

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    from itertools import tee

    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)
