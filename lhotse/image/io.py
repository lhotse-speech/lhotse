import io
import pickle
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Dict, List, Type, Union

import numpy as np

from lhotse.features.io import FileIO
from lhotse.image.image import Image
from lhotse.utils import Pathlike, is_module_available


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
    ) -> Image:
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
        assert is_module_available(
            "PIL"
        ), "In order to store images, please run 'pip install pillow'"
        import PIL.Image

        # Handle different types of input values
        if isinstance(value, PIL.Image.Image):
            img = np.array(value)
        elif isinstance(value, str):
            # It's a path, load the image first
            img = np.array(PIL.Image.open(value))
        elif isinstance(value, np.ndarray):
            # It's a numpy array
            img = value
        elif isinstance(value, bytes):
            # It's raw bytes, load it into PIL first to get dimensions
            img = np.array(PIL.Image.open(io.BytesIO(value)))
        else:
            raise ValueError(f"Unsupported image value type: {type(value)}")

        height, width = img.shape[:2]
        storage_key = self.write(key, img)

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
        self.io = FileIO(storage_path)

    @property
    def storage_path(self) -> str:
        return self.io.storage_path

    def read(self, key: str, as_pil_image: bool = False):
        """Read the image file and return it as a numpy array."""
        assert is_module_available(
            "PIL"
        ), "In order to load images, please run 'pip install pillow'"
        import PIL.Image

        with self.io.open_fileobj(key, "r") as (f, input_path):
            img = PIL.Image.open(f)
            img.load()  # trigger read, pillow has lazy loading
        if as_pil_image:
            return img
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
        self.io = FileIO(storage_path)

    @property
    def storage_path(self) -> str:
        return self.io.storage_path

    def write(self, key: str, value: np.ndarray) -> str:
        """Write a numpy array as an image file."""
        assert is_module_available(
            "PIL"
        ), "In order to store images, please run 'pip install pillow'"
        import PIL.Image

        # Use PNG as the default format if no extension is provided
        if not Path(key).suffix:
            key = key + ".png"

        # Convert numpy array to PIL Image and save
        img = PIL.Image.fromarray(value)
        with self.io.open_fileobj(key, "w", add_subdir=True) as (f, output_path):
            img.save(f)

        # Include sub-directory in the key, e.g. "abc/abcdef.png"
        if not self.io.is_url:
            return "/".join(Path(output_path).parts[-2:])
        else:
            return key


@register_reader
class PillowInMemoryReader(ImageReader):
    """
    Reads images from memory storage using Pillow.
    """

    name = "pillow_memory"

    def __init__(self, *args, **kwargs):
        pass

    def read(self, raw_data: bytes, as_pil_image: bool = False) -> np.ndarray:
        """Read the image from the bytes in memory."""
        assert is_module_available(
            "PIL"
        ), "In order to load images, please run 'pip install pillow'"
        import PIL.Image

        # Decode from pickle, which contains either bytes or numpy array
        data = pickle.loads(raw_data)

        # If it's bytes, load with PIL
        if isinstance(data, bytes):
            img = PIL.Image.open(io.BytesIO(data))
            if as_pil_image:
                return img
            return np.array(img)
        # If it's already a numpy array, return it directly
        elif isinstance(data, np.ndarray):
            if as_pil_image:
                return PIL.Image.fromarray(data)
            return data
        elif isinstance(data, PIL.Image.Image):
            if as_pil_image:
                return data
            return np.array(data)
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
