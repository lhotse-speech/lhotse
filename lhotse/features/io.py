from abc import ABCMeta, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import Type, Optional, List

import lilcom
import numpy as np

from lhotse.utils import Pathlike


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

    Each ``FeaturesWriter`` can also be used as a context manager, as some implementations
    might need to free a resource after the writing is finalized. By default nothing happens
    in the context manager functions, and this can be modified by the inheriting subclasses.

    Example:
        with MyWriter('some/path') as storage:
            extractor.extract_from_recording_and_store(recording, storage)

    The features loading must be defined separately in a class inheriting from ``FeaturesReader``.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def storage_path(self) -> str: ...

    @abstractmethod
    def write(self, key: str, value: np.ndarray) -> str: ...

    def __enter__(self): return self

    def __exit__(self, *args, **kwargs): ...


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
    def name(self) -> str: ...

    @abstractmethod
    def read(
            self,
            key: str,
            left_offset_frames: int = 0,
            right_offset_frames: Optional[int] = None
    ) -> np.ndarray: ...


READER_BACKENDS = {}
WRITER_BACKENDS = {}


def available_storage_backends() -> List[str]:
    return sorted(set(READER_BACKENDS).intersection(WRITER_BACKENDS))


def register_reader(cls):
    """
    Decorator used to add a new ``FeaturesReader`` to Lhotse's registry.

    Example:

        @register_reader
        class MyFeatureReader(FeatureReader):
            ...
    """
    READER_BACKENDS[cls.name] = cls
    return cls


def register_writer(cls):
    """
    Decorator used to add a new ``FeaturesWriter`` to Lhotse's registry.

    Example:

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
    name = 'lilcom_files'

    def __init__(self, storage_path: Pathlike, *args, **kwargs):
        super().__init__()
        self.storage_path = Path(storage_path)

    def read(
            self,
            key: str,
            left_offset_frames: int = 0,
            right_offset_frames: Optional[int] = None
    ) -> np.ndarray:
        with open(self.storage_path / key, 'rb') as f:
            arr = lilcom.decompress(f.read())
        return arr[left_offset_frames: right_offset_frames]


@register_writer
class LilcomFilesWriter(FeaturesWriter):
    """
    Writes Lilcom-compressed files to a directory on the local filesystem.
    ``storage_path`` corresponds to the directory path;
    ``storage_key`` for each utterance is the name of the file in that directory.
    """
    name = 'lilcom_files'

    def __init__(self, storage_path: Pathlike, tick_power: int = -5, *args, **kwargs):
        super().__init__()
        self.storage_path_ = Path(storage_path)
        self.storage_path_.mkdir(parents=True, exist_ok=True)
        self.tick_power = tick_power

    @property
    def storage_path(self) -> str:
        return self.storage_path_

    def write(self, key: str, value: np.ndarray) -> str:
        output_features_path = (self.storage_path_ / key).with_suffix('.llc')
        serialized_feats = lilcom.compress(value, tick_power=self.tick_power)
        with open(output_features_path, 'wb') as f:
            f.write(serialized_feats)
        return output_features_path.name


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
    name = 'numpy_files'

    def __init__(self, storage_path: Pathlike, *args, **kwargs):
        super().__init__()
        self.storage_path = Path(storage_path)

    def read(
            self,
            key: str,
            left_offset_frames: int = 0,
            right_offset_frames: Optional[int] = None
    ) -> np.ndarray:
        arr = np.load(self.storage_path / key, allow_pickle=False)
        return arr[left_offset_frames: right_offset_frames]


@register_writer
class NumpyFilesWriter(FeaturesWriter):
    """
    Writes non-compressed numpy arrays to files in a directory on the local filesystem.
    ``storage_path`` corresponds to the directory path;
    ``storage_key`` for each utterance is the name of the file in that directory.
    """
    name = 'numpy_files'

    def __init__(self, storage_path: Pathlike, *args, **kwargs):
        super().__init__()
        self.storage_path_ = Path(storage_path)
        self.storage_path_.mkdir(parents=True, exist_ok=True)

    @property
    def storage_path(self) -> str:
        return self.storage_path_

    def write(self, key: str, value: np.ndarray) -> str:
        output_features_path = (self.storage_path_ / key).with_suffix('.npy')
        np.save(output_features_path, value, allow_pickle=False)
        return output_features_path.name


"""
Non-compressed numpy arrays, stored in HDF5 file.
"""


@lru_cache(maxsize=None)
def lookup_cache_or_open(storage_path: str):
    """
    Helper internal function used in HDF5 readers.
    It opens the HDF files and keeps their handles open in a global program cache
    to avoid excessive amount of syscalls when the *Reader class is instantiated
    and destroyed in a loop repeatedly (frequent use-case).

    The file handles can be freed at any time by calling ``close_cached_file_handles()``.
    """
    import h5py
    return h5py.File(storage_path, 'r')


def close_cached_file_handles() -> None:
    """Closes the cached file handles in ``lookup_cache_or_open`` (see its docs for more details)."""
    lookup_cache_or_open.cache_clear()


@register_reader
class NumpyHdf5Reader(FeaturesReader):
    """
    Reads non-compressed numpy arrays from a HDF5 file with a "flat" layout.
    Each array is stored as a separate HDF ``Dataset`` because their shapes (numbers of frames) may vary.
    ``storage_path`` corresponds to the HDF5 file path;
    ``storage_key`` for each utterance is the key corresponding to the array (i.e. HDF5 "Group" name).
    """
    name = 'numpy_hdf5'

    def __init__(self, storage_path: Pathlike, *args, **kwargs):
        super().__init__()
        self.hdf = lookup_cache_or_open(storage_path)

    def read(
            self,
            key: str,
            left_offset_frames: int = 0,
            right_offset_frames: Optional[int] = None
    ) -> np.ndarray:
        # (pzelasko): If I understand HDF5/h5py correctly, this implementation reads only
        # the requested slice of the array into memory - but don't take my word for it.
        return self.hdf[key][left_offset_frames: right_offset_frames]


@register_writer
class NumpyHdf5Writer(FeaturesWriter):
    """
    Writes non-compressed numpy arrays to a HDF5 file with a "flat" layout.
    Each array is stored as a separate HDF ``Dataset`` because their shapes (numbers of frames) may vary.
    ``storage_path`` corresponds to the HDF5 file path;
    ``storage_key`` for each utterance is the key corresponding to the array (i.e. HDF5 "Group" name).
    """
    name = 'numpy_hdf5'

    def __init__(self, storage_path: Pathlike, *args, **kwargs):
        super().__init__()
        import h5py
        self.storage_path_ = storage_path
        self.hdf = h5py.File(storage_path, 'w')

    @property
    def storage_path(self) -> str:
        return self.storage_path_

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
    name = 'lilcom_hdf5'

    def __init__(self, storage_path: Pathlike, *args, **kwargs):
        super().__init__()
        self.hdf = lookup_cache_or_open(storage_path)

    def read(
            self,
            key: str,
            left_offset_frames: int = 0,
            right_offset_frames: Optional[int] = None
    ) -> np.ndarray:
        arr = lilcom.decompress(self.hdf[key].value.tobytes())
        return arr[left_offset_frames: right_offset_frames]


@register_writer
class LilcomHdf5Writer(FeaturesWriter):
    """
    Writes lilcom-compressed numpy arrays to a HDF5 file with a "flat" layout.
    Each array is stored as a separate HDF ``Dataset`` because their shapes (numbers of frames) may vary.
    ``storage_path`` corresponds to the HDF5 file path;
    ``storage_key`` for each utterance is the key corresponding to the array (i.e. HDF5 "Group" name).
    """
    name = 'lilcom_hdf5'

    def __init__(self, storage_path: Pathlike, tick_power: int = -5, *args, **kwargs):
        super().__init__()
        import h5py
        self.storage_path_ = storage_path
        self.hdf = h5py.File(storage_path, 'w')
        self.tick_power = tick_power

    @property
    def storage_path(self) -> str:
        return self.storage_path_

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
