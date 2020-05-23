from dataclasses import dataclass
from typing import Dict, List

from lhotse.features import Features
from lhotse.supervision import SupervisionSegment
from lhotse.utils import Seconds


# (PZ): After writing this, I'm thinking of making Cut an interface that might not necessarily expose
#  begin, end and channel (duration could stay I guess, ID I'm not sure about yet).
#  Then, there would be 2 (or more?) classes satisfying the interface: SingleCut and MultiCut.
#  That would make implementation of all the manipulations easier, I guess.

# I'm striving for maximally "lazy" implementation, e.g. when overlaying Cuts, I'd rather sum the feature matrices
#  only when somebody actually calls "load_features". Actually I might remove "features" from the interface too
#  for that reason.


@dataclass
class Cut:
    id: str

    # (PZ): I'm not super sure if this should be just one channel... but it's certainly easier to start this way
    channel: int

    # Begin and duration are needed to specify which chunk of features to load.
    begin: Seconds
    duration: Seconds

    supervisions: List[SupervisionSegment]

    # The features can span longer than the actual cut.
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

    def overlay(self, other: 'Cut', offset: Seconds = 0.0) -> 'Cut':
        """
        Return a new Cut that consists of supervisions of both input cuts,
        and feature matrix that consists of the sum of both cuts' feature matrices.
        """
        # TODO: "combination" of the feature matrices might be actually feature type dependent
        #  (e.g. for spectrograms we might want to just add, and
        #  for log-mel-energies we might want to exp first and log after)
        if self.duration < other.duration:
            # Simplification so that this function handles only the case when self is longer or equal to other.
            return other.overlay(self, offset=offset)
        raise NotImplementedError()

    def __add__(self, other: 'Cut') -> 'Cut':
        """Return a new Cut, with 'other' concatenated after 'self'."""
        # TODO: need to figure out a smart way to make them composable; maybe sth like class MultiCut...
        raise NotImplementedError()


@dataclass
class CutSet:
    cuts: Dict[str, Cut]
    # TODO: from_yaml, to_yaml, ...
