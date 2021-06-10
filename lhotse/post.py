# Copyright (c)  2021  Xiaomi Corp.       (author: Fangjun Kuang)

from dataclasses import dataclass

import numpy as np

from lhotse.features.io import FeaturesWriter, get_reader
from lhotse.utils import uuid4


@dataclass(order=True)
class Posteriors:
    """
    Represent the output of the neural network.
    It contains metadata about how it's stored: storage_type describes "how to read it", for now
    it supports numpy arrays serialized with np.save, as well as arrays compressed with lilcom;
    storage_path is the path to the file on the local filesystem.
    """
    num_frames: int

    num_dim: int  # model output dim

    # The subsampling factor of the model
    subsampling_factor: int

    # Storage type defines which features reader type should be instantiated
    # e.g. 'lilcom_files', 'numpy_files', 'lilcom_hdf5'
    storage_type: str

    # Storage path is either the path to some kind of archive (like HDF5 file) or a path
    # to a directory holding files with feature matrices (exact semantics depend on storage_type).
    storage_path: str

    # Storage key is either the key used to retrieve a feature matrix from an archive like HDF5,
    # or the name of the file in a directory (exact semantics depend on the storage_type).
    storage_key: str

    def load(self) -> np.ndarray:
        # noinspection PyArgumentList
        storage = get_reader(self.storage_type)(self.storage_path)

        # Load and return the posteriors from the storage
        return storage.read(self.storage_key)

    @staticmethod
    def from_dict(data: dict) -> 'Posteriors':
        return Posteriors(**data)


def save_posteriors(
        posts: np.ndarray,
        subsampling_factor: int,
        storage: FeaturesWriter,
) -> Posteriors:
    """
    Store ``posts`` array on disk.

    :param posts: a numpy ndarray (2-D) containing posteriors.
    :param: subsampling_factor: the subsampling factor of the model.
    :param storage: a ``FeaturesWriter`` object to use for array storage.
    :return: An instance of :class:`Posteriors`.
    """
    assert posts.ndim == 2

    posts_id = str(uuid4())
    storage_key = storage.write(posts_id, posts)

    return Posteriors(num_frames=posts.shape[0],
                      num_dim=posts.shape[1],
                      subsampling_factor=subsampling_factor,
                      storage_type=storage.name,
                      storage_path=str(storage.storage_path),
                      storage_key=storage_key)
