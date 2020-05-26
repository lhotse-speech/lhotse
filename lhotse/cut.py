from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional
from uuid import uuid4

import numpy as np

from lhotse.features import Features
from lhotse.supervision import SupervisionSegment
from lhotse.utils import Seconds, Decibels, overlaps, TimeSpan, overspans, Pathlike


# (PZ): After writing this, I'm thinking of making Cut an interface that might not necessarily expose
#  begin, end and channel (duration could stay I guess, ID I'm not sure about yet).
#  Then, there would be 2 (or more?) classes satisfying the interface: SingleCut and MultiCut.
#  That would make implementation of all the manipulations easier, I guess.

# I'm striving for maximally "lazy" implementation, e.g. when overlaying Cuts, I'd rather sum the feature matrices
#  only when somebody actually calls "load_features". Actually I might remove "features" from the interface too
#  for that reason.


class Cut(metaclass=ABCMeta):
    """
    A Cut is a single segment that we'll train on. It contains the features corresponding to
    a piece of a recording, with zero or more SupervisionSegments.
    """

    @abstractmethod
    @property
    def begin(self) -> Seconds:
        pass

    @abstractmethod
    @property
    def duration(self) -> Seconds:
        pass

    @abstractmethod
    @property
    def end(self) -> Seconds:
        pass

    @abstractmethod
    def load_features(self, root_dir: Optional[Pathlike] = None) -> np.ndarray:
        """Load the features corresponding to this Cut into numpy array."""
        pass

    @abstractmethod
    def truncate(self, offset: Seconds = 0.0, duration: Seconds = 0.0) -> 'Cut':
        """Return a new Cut with shorter time span."""
        pass

    @abstractmethod
    def overlay(self, other: 'Cut', offset: Seconds = 0.0, snr: Decibels = 0.0) -> 'Cut':
        """
        Return a new Cut that consists of supervisions of both input cuts,
        and feature matrix that consists of the sum of both cuts' feature matrices.
        The 'other' Cut magnitude is scaled to meet the SNR requirement specified by 'snr'.
        """
        pass

    @abstractmethod
    def __add__(self, other: 'Cut', snr: float) -> 'Cut':
        """Return a new Cut, with 'other' concatenated after 'self'."""
        pass


@dataclass
class SingleCut(Cut):
    """
    A Cut is a single segment that we'll train on. It contains the features corresponding to
    a piece of a recording, with zero or more SupervisionSegments.
    """

    def end(self) -> Seconds:
        return self.begin + self.duration

    def load_features(self, root_dir: Optional[Pathlike] = None) -> np.ndarray:
        return self.features.load(
            root_dir=root_dir,
            start=self.begin,
            duration=self.duration,
            # TODO: Oops. Need to move frame shift/length info into the Features object
            frame_length=None,
            frame_shift=None
        )

    def truncate(
            self,
            offset: Seconds = 0.0,
            duration: Seconds = 0.0,
            keep_excessive_supervisions: bool = True
    ) -> 'Cut':
        new_begin = self.begin + offset
        new_duration = self.duration if duration <= 0.0 else duration
        new_time_span = TimeSpan(begin=new_begin, end=new_begin + new_duration)
        criterion = overlaps if keep_excessive_supervisions else overspans
        return SingleCut(
            id=str(uuid4()),
            channel=self.channel,
            begin=new_begin,
            duration=new_duration,
            supervisions=[
                segment for segment in self.supervisions if criterion(new_time_span, segment)
            ],
            features=self.features
        )

    def overlay(self, other: 'Cut', offset: Seconds = 0.0, snr: Decibels = 0.0) -> 'Cut':
        pass

    def __add__(self, other: 'Cut', snr: float) -> 'Cut':
        pass

    id: str

    # (PZ): I'm not super sure if this should be just one channel... but it's certainly easier to start this way
    channel: int

    # Begin and duration are needed to specify which chunk of features to load.
    begin: Seconds
    duration: Seconds

    # Supervisions that will be used as targets for model training later on. They don't have to cover the whole
    # cut duration. They also might overlap.
    supervisions: List[SupervisionSegment]

    # The features can span longer than the actual cut - the Features object "knows" its start and end time
    # within the underlying recording. We can expect the interval [begin, begin + duration] to be a subset of the
    # interval represented in features.
    features: Features


@dataclass
class MultiCut(Cut):
    underlying_cuts: List[Cut]


@dataclass
class __Cut:
    """
    A Cut is a single segment that we'll train on. It contains the features corresponding to
    a piece of a recording, with zero or more SupervisionSegments.
    """
    id: str

    # (PZ): I'm not super sure if this should be just one channel... but it's certainly easier to start this way
    channel: int

    # Begin and duration are needed to specify which chunk of features to load.
    begin: Seconds
    duration: Seconds

    # Supervisions that will be used as targets for model training later on. They don't have to cover the whole
    # cut duration. They also might overlap.
    supervisions: List[SupervisionSegment]

    # The features can span longer than the actual cut - the Features object "knows" its start and end time
    # within the underlying recording. We can expect the interval [begin, begin + duration] to be a subset of the
    # interval represented in features.
    features: Features

    @property
    def end(self):
        return self.begin + self.duration

    def truncate(self, offset: Seconds = 0.0, duration: Seconds = 0.0) -> 'Cut':
        """Return a new Cut with shorter time span."""
        # TODO: there will be supervisions that might get "cut" - should we discard them or keep them?
        #  Both options seem suboptimal - maybe we should raise an exception instead?
        #  Or all of those, but controllable with flags...
        raise NotImplementedError()

    def overlay(self, other: 'Cut', offset: Seconds = 0.0, snr: Decibels = 0.0) -> 'Cut':
        """
        Return a new Cut that consists of supervisions of both input cuts,
        and feature matrix that consists of the sum of both cuts' feature matrices.
        The 'other' Cut magnitude is scaled to meet the SNR requirement specified by 'snr'.
        """
        # TODO: "combination" of the feature matrices might be actually feature type dependent
        #  (e.g. for spectrograms we might want to just add, and
        #  for log-mel-energies we might want to exp first and log after)
        if self.duration < other.duration:
            # Simplification so that this function handles only the case when self is longer or equal to other.
            # Notice the SNR is flipped...
            return other.overlay(self, offset=offset, snr=-snr)
        raise NotImplementedError()

    def __add__(self, other: 'Cut', snr: float) -> 'Cut':
        """Return a new Cut, with 'other' concatenated after 'self'."""
        # TODO: need to figure out a smart way to make them composable; maybe sth like class MultiCut...
        raise NotImplementedError()


@dataclass
class CutSet:
    cuts: Dict[str, Cut]
    # TODO: from_yaml, to_yaml, ...
