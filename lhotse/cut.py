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


@dataclass
class Cut:
    """
    A Cut is a single segment that we'll train on. It contains the features corresponding to
    a piece of a recording, with zero or more SupervisionSegments.
    """
    id: str

    # (PZ): I'm not super sure if this should be just one channel... but it's certainly easier to start this way
    channel: int

    # Begin and duration are needed to specify which chunk of features to load.
    start: Seconds
    duration: Seconds

    # The features can span longer than the actual cut - the Features object "knows" its start and end time
    # within the underlying recording. We can expect the interval [begin, begin + duration] to be a subset of the
    # interval represented in features.
    features: Features

    # Supervisions that will be used as targets for model training later on. They don't have to cover the whole
    # cut duration. They also might overlap.
    supervisions: List[SupervisionSegment]

    def end(self) -> Seconds:
        return self.start + self.duration

    def load_features(self, root_dir: Optional[Pathlike] = None) -> np.ndarray:
        return self.features.load(root_dir=root_dir, start=self.start, duration=self.duration)

    def supervisions(self) -> Dict[str, np.ndarray]:
        pass

    def truncate(
            self,
            *,
            offset: Seconds = 0.0,
            until: Optional[Seconds] = None,
            keep_excessive_supervisions: bool = True
    ) -> 'Cut':
        new_start = self.start + offset
        new_duration = self.duration if until is None else until
        assert new_duration > 0.0
        new_time_span = TimeSpan(start=new_start, end=new_duration)
        criterion = overlaps if keep_excessive_supervisions else overspans
        return Cut(
            id=str(uuid4()),
            channel=self.channel,
            start=new_start,
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


@dataclass
class CutSet:
    cuts: Dict[str, Cut]

    # TODO: from_yaml, to_yaml, ...

    def __len__(self) -> int:
        return len(self.cuts)
