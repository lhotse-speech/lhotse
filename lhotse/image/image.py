from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from lhotse.utils import Pathlike, fastcopy


@dataclass
class Image:
    """
    The Image manifest describes an image that is stored somewhere: it might be
    in a file, in memory, in the cloud, etc.
    Image helps abstract away from the actual storage mechanism and location by
    providing a method called :meth:`.Image.load`.

    Image manifest can be easily created by calling
    :meth:`lhotse.image.io.PillowWriter.store_image`, for example::

        >>> from lhotse.image.io import PillowWriter
        >>> with PillowWriter('images/') as writer:
        ...     manifest = writer.store_image('image-1', 'path/to/image.jpg')
    """

    # Storage type defines which image reader type should be instantiated
    # e.g. 'pillow_files', 'pillow_memory'
    storage_type: str

    # Storage path is either the path to a directory holding files with images
    # or None for in-memory images
    storage_path: str

    # Storage key is the name of the file in a directory
    # or binary data for in-memory images
    storage_key: str

    # Width and height of the image in pixels
    width: int
    height: int

    @property
    def shape(self) -> Tuple[int, int]:
        """Return the shape of the image as (height, width)."""
        return (self.height, self.width)

    @property
    def is_in_memory(self) -> bool:
        """Check if the image is stored in memory."""
        # Import locally to avoid circular dependencies
        from lhotse.image.io import is_in_memory

        return is_in_memory(self.storage_type)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Image":
        """Create an Image from a dictionary representation."""
        if (
            "storage_key" in data
            and "storage_type" in data
            and "storage_path" not in data
        ):
            data["storage_path"] = None
        return cls(**data)

    def load(self) -> np.ndarray:
        """
        Load the image from the underlying storage.
        Returns a numpy array with shape (height, width, channels).
        """
        # Import locally to avoid circular dependencies
        from lhotse.image.io import get_reader

        # noinspection PyArgumentList
        storage = get_reader(self.storage_type)(self.storage_path)
        # Load and return the image from the storage
        return storage.read(self.storage_key)

    def with_path_prefix(self, path: Pathlike) -> "Image":
        """
        Return a copy of the image with ``path`` added as a prefix
        to the ``storage_path`` member.
        """
        return fastcopy(self, storage_path=str(Path(path) / self.storage_path))

    def move_to_memory(self) -> "Image":
        """
        Return a copy of the image stored in memory.
        """
        # Import locally to avoid circular dependencies
        from lhotse.image.io import get_memory_writer

        if self.storage_type in ("pillow_memory"):
            return self  # nothing to do

        img = self.load()
        writer = get_memory_writer("pillow_memory")()
        data = writer.write("", img)  # key is ignored by in memory writers
        return Image(
            storage_type=writer.name,
            storage_key=data,
            storage_path="",
            width=self.width,
            height=self.height,
        )

    def __repr__(self):
        return (
            f"Image("
            f"storage_type='{self.storage_type}', "
            f"storage_path='{self.storage_path}', "
            f"storage_key='{self.storage_key if isinstance(self.storage_key, str) else '<binary-data>'}', "
            f"width={self.width}, "
            f"height={self.height}"
            f")"
        )
