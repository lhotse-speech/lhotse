import decimal
from dataclasses import dataclass
from math import isclose
from typing import List, Optional

import numpy as np

from lhotse import FeaturesWriter
from lhotse.features.io import get_reader
from lhotse.utils import Seconds, asdict_nonull, rich_exception_info


@dataclass
class Array:
    """ """

    # Storage type defines which features reader type should be instantiated
    # e.g. 'lilcom_files', 'numpy_files', 'lilcom_hdf5'
    storage_type: str

    # Storage path is either the path to some kind of archive (like HDF5 file) or a path
    # to a directory holding files with feature matrices (exact semantics depend on storage_type).
    storage_path: str

    # Storage key is either the key used to retrieve an array from an archive like HDF5,
    # or the name of the file in a directory (exact semantics depend on the storage_type).
    storage_key: str

    # Shape of the array once loaded into memory.
    shape: List[int]

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def to_dict(self) -> dict:
        return asdict_nonull(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Array":
        return cls(**data)

    @classmethod
    def store(cls, key: str, value: np.ndarray, writer: FeaturesWriter) -> "Array":
        storage_key = writer.write(key, value)
        return Array(
            storage_type=writer.name,
            storage_path=str(writer.storage_path),
            storage_key=storage_key,
            shape=list(value.shape),
        )

    @rich_exception_info
    def load(self) -> np.ndarray:

        # noinspection PyArgumentList
        storage = get_reader(self.storage_type)(self.storage_path)

        # Load and return the array from the storage
        return storage.read(self.storage_key)


@dataclass
class TemporalArray:
    """ """

    # Manifest describing the base array.
    array: Array

    # Indicates which dim corresponds to the time dimension:
    # e.g., PCM audio samples indexes, feature frame indexes, chunks indexes, etc.
    temporal_dim: int

    # The time interval (in seconds, or fraction of a second) between the start timestamps
    # of consecutive frames. Only defined when ``temporal_axis`` is not ``None``.
    frame_shift: Seconds

    # Information about the time range of the features.
    # We only need to specify start, as duration can be computed from
    # the shape, temporal_axis, and frame_shift.
    start: Seconds

    @property
    def shape(self):
        return self.array.shape

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def duration(self) -> Optional[Seconds]:
        return self.shape[self.temporal_dim] * self.frame_shift

    @property
    def end(self) -> Optional[Seconds]:
        return self.start + self.duration

    def to_dict(self) -> dict:
        return asdict_nonull(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TemporalArray":
        array = Array.from_dict(data.pop('array'))
        return cls(array=array, **data)

    @classmethod
    def store(
        cls,
        key: str,
        value: np.ndarray,
        writer: FeaturesWriter,
        frame_shift: Seconds,
        temporal_dim: int = 0,
        start: Seconds = 0,
    ) -> "TemporalArray":
        return TemporalArray(
            array=Array.store(key=key, value=value, writer=writer),
            temporal_dim=temporal_dim,
            frame_shift=frame_shift,
            start=start
        )

    # @rich_exception_info
    def load(
        self,
        start: Optional[Seconds] = None,
        duration: Optional[Seconds] = None,
    ) -> np.ndarray:

        # noinspection PyArgumentList
        storage = get_reader(self.array.storage_type)(self.array.storage_path)
        left_offset_frames, right_offset_frames = 0, None

        if start is None:
            start = self.start
        # In case the caller requested only a sub-span of the features, trim them.
        # Left trim
        if start < self.start - 1e-5:
            raise ValueError(
                f"Cannot load array starting from {start}s. "
                f"The available range is ({self.start}, {self.end}) seconds."
            )
        if not isclose(start, self.start):
            left_offset_frames = seconds_to_frames(
                start - self.start,
                frame_shift=self.frame_shift,
                max_index=self.shape[self.temporal_dim],
            )
        # Right trim
        if duration is not None:
            right_offset_frames = left_offset_frames + seconds_to_frames(
                duration,
                frame_shift=self.frame_shift,
                max_index=self.shape[self.temporal_dim],
            )

        # Load and return the features (subset) from the storage
        return storage.read(
            self.array.storage_key,
            left_offset_frames=left_offset_frames,
            right_offset_frames=right_offset_frames,
        )


def seconds_to_frames(duration: Seconds, frame_shift: Seconds, max_index: int) -> int:
    """
    Convert time quantity in seconds to a frame index.
    It takes the shape of the array into account and limits
    the possible indices values to be compatible with the shape.
    """
    assert duration >= 0
    index = int(
        decimal.Decimal(
            # 8 is a good number because cases like 14.49175 still work correctly,
            # while problematic cases like 14.49999999998 are typically breaking much later than 8th decimal
            # with double-precision floats.
            round(duration / frame_shift, ndigits=8)
        ).quantize(0, rounding=decimal.ROUND_HALF_UP)
    )
    return min(index, max_index)
