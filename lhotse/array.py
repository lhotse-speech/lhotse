from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from lhotse.utils import Seconds, asdict_nonull


@dataclass
class Array:
    """ """

    shape: List[int]

    @property
    def num_axes(self) -> int:
        return len(self.shape)

    # Storage type defines which features reader type should be instantiated
    # e.g. 'lilcom_files', 'numpy_files', 'lilcom_hdf5'
    storage_type: str

    # Storage path is either the path to some kind of archive (like HDF5 file) or a path
    # to a directory holding files with feature matrices (exact semantics depend on storage_type).
    storage_path: str

    # Storage key is either the key used to retrieve a feautre matrix from an archive like HDF5,
    # or the name of the file in a directory (exact semantics depend on the storage_type).
    storage_key: str

    temporal_axis: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict_nonull(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Array":
        return cls(**data)

    def load(
        self,
        start: Optional[Seconds] = None,
        duration: Optional[Seconds] = None,
    ) -> np.ndarray:
        pass
