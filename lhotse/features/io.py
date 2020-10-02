from abc import ABCMeta, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import Type

import lilcom
import numpy as np

from lhotse.utils import Pathlike


class FeaturesWriter(metaclass=ABCMeta):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def storage_path(self) -> str: ...

    @abstractmethod
    def write(self, key: str, value: np.ndarray) -> str: ...


class FeaturesReader(metaclass=ABCMeta):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def read(self, key: str) -> np.ndarray: ...


READER_BACKENDS = {}
WRITER_BACKENDS = {}


def register_reader(cls):
    READER_BACKENDS[cls.name] = cls
    return cls


def register_writer(cls):
    WRITER_BACKENDS[cls.name] = cls
    return cls


def get_reader(name: str) -> Type[FeaturesReader]:
    return READER_BACKENDS.get(name)


def get_writer(name: str) -> Type[FeaturesWriter]:
    return WRITER_BACKENDS.get(name)


@register_reader
class LilcomFilesReader(FeaturesReader):
    name = 'lilcom_files'

    def __init__(self, storage_path: Pathlike, *args, **kwargs):
        super().__init__()
        self.storage_path = Path(storage_path)

    def read(self, key: str) -> np.ndarray:
        """Expects key to be a valid filesystem path."""
        with open(self.storage_path / key, 'rb') as f:
            return lilcom.decompress(f.read())


@register_writer
class LilcomFilesWriter(FeaturesWriter):
    name = 'lilcom_files'

    def __init__(self, storage_path: Pathlike, tick_power: int = -5, *args, **kwargs):
        """storage_path will point to a directory"""
        super().__init__()
        self.storage_path_ = Path(storage_path)
        self.storage_path_.mkdir(parents=True, exist_ok=True)
        self.tick_power = tick_power

    @property
    def storage_path(self) -> str:
        return self.storage_path_

    def write(self, key: str, value: np.ndarray) -> str:
        """Expects key to be the ID of the features object."""
        output_features_path = (self.storage_path_ / key).with_suffix('.llc')
        serialized_feats = lilcom.compress(value, tick_power=self.tick_power)
        with open(output_features_path, 'wb') as f:
            f.write(serialized_feats)
        return output_features_path.name


@register_reader
class NumpyFilesReader(FeaturesReader):
    name = 'numpy_files'

    def __init__(self, storage_path: Pathlike, *args, **kwargs):
        super().__init__()
        self.storage_path = Path(storage_path)

    def read(self, key: str) -> np.ndarray:
        """Expects key to be a valid filesystem path."""
        return np.load(self.storage_path / key, allow_pickle=False)


@register_writer
class NumpyFilesWriter(FeaturesWriter):
    name = 'numpy_files'

    def __init__(self, storage_path: Pathlike, *args, **kwargs):
        """storage_path will point to a directory"""
        super().__init__()
        self.storage_path_ = Path(storage_path)
        self.storage_path_.mkdir(parents=True, exist_ok=True)

    @property
    def storage_path(self) -> str:
        return self.storage_path_

    def write(self, key: str, value: np.ndarray) -> str:
        """Expects key to be the ID of the features object."""
        output_features_path = (self.storage_path_ / key).with_suffix('.npy')
        np.save(output_features_path, value, allow_pickle=False)
        return output_features_path.name


@lru_cache(maxsize=10)
def lookup_cache_or_open(storage_path: str):
    import h5py
    return h5py.File(storage_path, 'r')


def close_cached_hdf_files() -> None:
    lookup_cache_or_open.cache_clear()


@register_reader
class LilcomHdf5Reader(FeaturesReader):
    name = 'lilcom_hdf5'

    def __init__(self, storage_path: Pathlike, *args, **kwargs):
        super().__init__()
        self.hdf = lookup_cache_or_open(storage_path)

    def read(self, key: str) -> np.ndarray:
        return lilcom.decompress(self.hdf[key].value.tobytes())


@register_writer
class LilcomHdf5Writer(FeaturesWriter):
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
