import io
import pickle
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Dict, List, Type, Union

import numpy as np

from lhotse.image.image import Image
from lhotse.utils import Pathlike


class ImageReader(metaclass=ABCMeta):
    """
    ``ImageReader`` defines the interface of how to load images from a particular storage backend.
    This backend could either be:

    - files on a local filesystem;
    - in-memory storage;
    - cloud storage;
    - etc.

    Each class inheriting from ``ImageReader`` must define:

    - the ``read()`` method, which defines the loading operation
      (accepts the ``key`` to locate the image in the storage and return it).
    - the ``name()`` property that is unique to this particular storage mechanism.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def read(
        self,
        key: str,
    ) -> np.ndarray:
        ...


class ImageWriter(metaclass=ABCMeta):
    """
    ``ImageWriter`` defines the interface of how to store images in a particular storage backend.
    This backend could either be:

    - files on a local filesystem;
    - in-memory storage;
    - cloud storage;
    - etc.

    Each class inheriting from ``ImageWriter`` must define:

    - the ``write()`` method, which defines the storing operation
      (accepts a ``key`` used to place the ``value`` image in the storage);
    - the ``storage_path()`` property, which is either a directory for the files,
      or empty string for in-memory storage.
    - the ``name()`` property that is unique to this particular storage mechanism.
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

    def store_image(
        self,
        key: str,
        value: Union[str, np.ndarray, bytes],
    ) -> "Image":
        """
        Store an image in the underlying storage and return a manifest
        describing how to retrieve it.

        :param key: An ID that uniquely identifies the image.
        :param value: The image to be stored. Can be:
            - A path to an image file
            - A numpy array with shape (height, width, channels)
            - Raw bytes of an image file
        :return: A manifest of type :class:`~lhotse.image.Image`
        """
        # Import Image here to avoid circular imports
        from lhotse.image.image import Image

        # Handle different types of input values
        if isinstance(value, str):
            # It's a path, load the image first
            import PIL.Image

            img = np.array(PIL.Image.open(value))
            height, width = img.shape[:2]
            storage_key = self.write(key, img)
        elif isinstance(value, np.ndarray):
            # It's a numpy array
            height, width = value.shape[:2]
            storage_key = self.write(key, value)
        elif isinstance(value, bytes):
            # It's raw bytes, load it into PIL first to get dimensions
            import PIL.Image

            img = np.array(PIL.Image.open(io.BytesIO(value)))
            height, width = img.shape[:2]
            storage_key = self.write(key, img)
        else:
            raise ValueError(f"Unsupported image value type: {type(value)}")

        return Image(
            storage_type=self.name,
            storage_path=self.storage_path,
            storage_key=storage_key,
            width=width,
            height=height,
        )

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        ...


READER_BACKENDS = {}
WRITER_BACKENDS = {}


def available_storage_backends() -> List[str]:
    return sorted(set(READER_BACKENDS).intersection(WRITER_BACKENDS))


def register_reader(cls):
    """
    Decorator used to add a new ``ImageReader`` to Lhotse's registry.
    """
    READER_BACKENDS[cls.name] = cls
    return cls


def register_writer(cls):
    """
    Decorator used to add a new ``ImageWriter`` to Lhotse's registry.
    """
    WRITER_BACKENDS[cls.name] = cls
    return cls


def get_reader(name: str) -> Type[ImageReader]:
    """
    Find the available ``ImageReader`` implementation with the provided name.
    The name is the value of the ``name`` property in a concrete ``ImageReader`` implementation.
    """
    if name not in READER_BACKENDS:
        raise ValueError(
            f"No image reader backend with name '{name}' is available. "
            f"Available backends: {list(READER_BACKENDS.keys())}"
        )
    return READER_BACKENDS[name]


def get_writer(name: str) -> Type[ImageWriter]:
    """
    Find the available ``ImageWriter`` implementation with the provided name.
    The name is the value of the ``name`` property in a concrete ``ImageWriter`` implementation.
    """
    if name not in WRITER_BACKENDS:
        raise ValueError(
            f"No image writer backend with name '{name}' is available. "
            f"Available backends: {list(WRITER_BACKENDS.keys())}"
        )
    return WRITER_BACKENDS[name]


def get_memory_writer(name: str) -> Type[ImageWriter]:
    """
    Find the available in-memory ``ImageWriter`` implementation with the provided name.
    """
    writer = get_writer(name)
    assert is_in_memory(name), f"Storage type '{name}' is not an in-memory storage."
    return writer


def is_in_memory(storage_type: str) -> bool:
    """
    Determines whether a given storage_type represents an in-memory storage.
    """
    return storage_type == "pillow_memory"


@register_reader
class PillowReader(ImageReader):
    """
    Reads image files using Pillow from a directory on the local filesystem.
    ``storage_path`` corresponds to the directory path;
    ``storage_key`` for each image is the name of the file in that directory.
    """

    name = "pillow_files"

    def __init__(self, storage_path: Pathlike, *args, **kwargs):
        super().__init__()
        self.storage_path = Path(storage_path)

    def read(self, key: str) -> np.ndarray:
        """Read the image file and return it as a numpy array."""
        try:
            import PIL.Image
        except ImportError:
            raise ImportError("To use PillowReader, please 'pip install pillow' first.")

        img_path = self.storage_path / key
        with PIL.Image.open(img_path) as img:
            # Convert to numpy array (height, width, channels)
            return np.array(img)


@register_writer
class PillowWriter(ImageWriter):
    """
    Writes image files using Pillow to a directory on the local filesystem.
    ``storage_path`` corresponds to the directory path;
    ``storage_key`` for each image is the name of the file in that directory.
    """

    name = "pillow_files"

    def __init__(self, storage_path: Pathlike, *args, **kwargs):
        super().__init__()
        self.storage_path_ = Path(storage_path)
        self.storage_path_.mkdir(parents=True, exist_ok=True)

    @property
    def storage_path(self) -> str:
        return str(self.storage_path_)

    def write(self, key: str, value: np.ndarray) -> str:
        """Write a numpy array as an image file."""
        try:
            import PIL.Image
        except ImportError:
            raise ImportError("To use PillowWriter, please 'pip install pillow' first.")

        # Introduce a sub-directory that starts with the first 3 characters of the key
        subdir = self.storage_path_ / key[:3]
        subdir.mkdir(exist_ok=True)

        p = subdir / key
        # Use PNG as the default format if no extension is provided
        if not p.suffix:
            p = p.with_suffix(".png")

        # Convert numpy array to PIL Image and save
        img = PIL.Image.fromarray(value)
        img.save(p)

        # Include sub-directory in the key, e.g. "abc/abcdef.png"
        return "/".join(p.parts[-2:])


@register_reader
class PillowInMemoryReader(ImageReader):
    """
    Reads images from memory storage using Pillow.
    """

    name = "pillow_memory"

    def __init__(self, *args, **kwargs):
        pass

    def read(self, raw_data: bytes) -> np.ndarray:
        """Read the image from the bytes in memory."""
        try:
            import PIL.Image
        except ImportError:
            raise ImportError(
                "To use PillowInMemoryReader, please 'pip install pillow' first."
            )

        # Decode from pickle, which contains either bytes or numpy array
        data = pickle.loads(raw_data)

        # If it's bytes, load with PIL
        if isinstance(data, bytes):
            img = PIL.Image.open(io.BytesIO(data))
            return np.array(img)
        # If it's already a numpy array, return it directly
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")


@register_writer
class PillowInMemoryWriter(ImageWriter):
    """
    Writes images to memory storage using Pillow.
    """

    name = "pillow_memory"

    def __init__(self, *args, **kwargs):
        pass

    @property
    def storage_path(self) -> None:
        return None

    def write(self, key: str, value: np.ndarray) -> bytes:
        """
        Store the image in memory. We're simply pickle-ing the numpy array.
        """
        return pickle.dumps(value)

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
