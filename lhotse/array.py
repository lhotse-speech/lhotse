import decimal
from dataclasses import asdict, dataclass
from math import isclose
from typing import List, Optional, Union

import numpy as np

from lhotse.utils import Seconds


@dataclass
class Array:
    """
    The Array manifest describes a numpy array that is stored somewhere: it might be
    in an HDF5 file, a compressed numpy file, on disk, in the cloud, etc.
    Array helps abstract away from the actual storage mechanism and location by
    providing a method called :meth:`.Array.load`.

    We don't assume anything specific about the array itself: it might be
    a feature matrix, an embedding, network activations, posteriors, alignment, etc.
    However, if the array has a temporal component, it is better to use the
    :class:`.TemporalArray` manifest instead.

    Array manifest can be easily created by calling
    :meth:`lhotse.features.io.FeaturesWriter.store_array`, for example::

        >>> from lhotse import NumpyHdf5Writer
        >>> ivector = np.random.rand(300)
        >>> with NumpyHdf5Writer('ivectors.h5') as writer:
        ...     manifest = writer.store_array('ivec-1', ivector)
    """

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
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Array":
        return cls(**data)

    def load(self) -> np.ndarray:
        """
        Load the array from the underlying storage.
        """
        from lhotse.features.io import get_reader

        # noinspection PyArgumentList
        storage = get_reader(self.storage_type)(self.storage_path)
        # Load and return the array from the storage
        return storage.read(self.storage_key)


@dataclass
class TemporalArray:
    """
    The :class:`.TemporalArray` manifest describes a numpy array that is stored somewhere:
    it might be in an HDF5 file, a compressed numpy file, on disk, in the cloud, etc.
    Like :class:`.Array`, it helps abstract away from the actual storage mechanism
    and location by providing a method called :meth:`.TemporalArray.load`.

    Unlike with :class:`.Array`, we assume that the array has a temporal dimension.
    It allows us to perform partial reads for sub-segments of the data if the underlying
    ``storage_type`` allows that.

    :class:`.TemporalArray` manifest can be easily created by calling
    :meth:`lhotse.features.io.FeaturesWriter.store_array` and specifying arguments
    related to its temporal nature; for example::

        >>> from lhotse import NumpyHdf5Writer
        >>> alignment = np.random.randint(500, size=131)
        >>> assert alignment.shape == (131,)
        >>> with NumpyHdf5Writer('alignments.h5') as writer:
        ...     manifest = writer.store_array(
        ...         key='ali-1',
        ...         value=alignment,
        ...         frame_shift=0.4,  # e.g., 10ms frames and subsampling_factor=4
        ...         temporal_dim=0,
        ...         start=0
        ...     )
    """

    # Manifest describing the base array.
    array: Array

    # Indicates which dim corresponds to the time dimension:
    # e.g., PCM audio samples indexes, feature frame indexes, chunks indexes, etc.
    temporal_dim: int

    # The time interval (in seconds, or fraction of a second) between the start timestamps
    # of consecutive frames.
    frame_shift: Seconds

    # Information about the time range of the features.
    # We only need to specify start, as duration can be computed from
    # the shape, temporal_dim, and frame_shift.
    start: Seconds

    @property
    def shape(self) -> List[int]:
        return self.array.shape

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def duration(self) -> Seconds:
        return self.shape[self.temporal_dim] * self.frame_shift

    @property
    def end(self) -> Seconds:
        return self.start + self.duration

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TemporalArray":
        array = Array.from_dict(data.pop("array"))
        return cls(array=array, **data)

    def load(
        self,
        start: Optional[Seconds] = None,
        duration: Optional[Seconds] = None,
    ) -> np.ndarray:
        """
        Load the array from the underlying storage.
        Optionally perform a partial read along the ``temporal_dim``.

        :param start: when specified, we'll offset the read by ``start`` after
            converting it to a number of frames based on ``self.frame_shift``.
        :param duration: when specified, we'll limit the read to a number of
            frames equivalent to ``duration`` under ``self.frame_shift``.
        :return: A numpy array or a relevant slice of it.
        """
        from lhotse.features.io import get_reader

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


def deserialize_array(raw_data: dict) -> Union[Array, TemporalArray]:
    """
    Figures out the right manifest type to use for deserialization.

    :param raw_data: The result of calling ``.to_dict`` on :class:`.Array`
        or :class:`.TemporalArray`.
    :return an :class:`.Array.` or :class:`.TemporalArray` instance.
    """
    if "array" in raw_data:
        return TemporalArray.from_dict(raw_data)
    if "shape" in raw_data:
        return Array.from_dict(raw_data)
    raise ValueError(f"Cannot deserialize array from: {raw_data}")
