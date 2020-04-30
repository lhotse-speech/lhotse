"""
This file is just a rough sketch for now.
"""
from dataclasses import dataclass
from typing import Union, List, Optional, Iterable, Any

import numpy as np

from lhotse.audio import AudioSet
from lhotse.supervision import SupervisionSet
from lhotse.utils import Seconds


@dataclass
class Features:
    """Represents features extracted for some particular time range in a given recording and channel."""
    recording_id: str
    channel_id: int
    start: Seconds
    duration: Seconds

    # Variant A: path to a single-segment lilcom archive serialized in some storage
    storage_path: str

    # Variant B: path to a multi-segment lilcom archive serialized in some storage AND key to access it
    storage_path: str
    storage_key: str

    # When created from SupervisionSet maybe it makes sense to remember where this comes from... or not...
    # if we limited ourselves to just using SupervisionSet, it could serve as the "storage_key"
    supervision_id: Optional[str] = None

    def load(self) -> np.ndarray:
        pass


@dataclass
class FeatureSet:
    """
    Represents a feature manifest, and allows to read features for given recordings
    within particular channels and time ranges.
    When a given recording/time-range/channel is unavailable, raises a KeyError.
    """
    # TODO(pzelasko): we might need some efficient indexing structure,
    #                 e.g. Dict[<recording-id>, Dict[<channel-id>, IntervalTree]] (pip install intervaltree)
    features: List[Features]

    @staticmethod
    def from_yaml(param):
        pass

    @staticmethod
    def to_yaml(param):
        pass

    def load(
            self,
            recording_id: str,
            channel_id: int,
            start: Seconds,
            duration: Seconds,
    ) -> np.ndarray:
        # raise a KeyError when any of the requirements is not met
        pass

    def __iter__(self) -> Iterable[Features]:
        return iter(self.features)

    def __len__(self) -> int:
        return len(self.features)


@dataclass
class FeatureSegment:
    """
    PZ: Now that I've written it, seems redundant with SupervisionSegment... maybe it should be a base class for it
    """
    recording_id: str
    channel: int
    start: Seconds
    duration: Seconds


class FeatureExtractor:
    def with_audio_set(self, audio_set: AudioSet) -> 'FeatureExtractor':
        self.audio_set = audio_set
        return self

    def with_augmentation(self, augmentation: Any) -> 'FeatureExtractor':
        self.augmentation = augmentation
        return self

    def with_algorithm(
            self,
            method='mfcc',
            frame_size: Seconds = 0.025,
            frame_shift: Seconds = 0.01,
            **basically_a_lot_of_other_params
    ) -> 'FeatureExtractor':
        self.algorithm = None  # TODO
        return self

    def with_segmentation(
            self,
            segmentation: Union[SupervisionSet, List[FeatureSegment]],
            extra_left_seconds: Seconds = 0.0,
            extra_right_seconds: Seconds = 0.0
    ) -> 'FeatureExtractor':
        self.segmentation = segmentation
        self.extra_left_seconds = extra_left_seconds
        self.extra_right_seconds = extra_right_seconds
        return self

    def extract(self) -> FeatureSet:
        # TODO: the juicy stuff (librosa, torchaudio, eventually also augmentation etc.) goes here
        return FeatureSet()
