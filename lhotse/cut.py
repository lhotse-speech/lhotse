from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Iterable, Any, Union
from uuid import uuid4

import numpy as np
import yaml

from lhotse.features import Features, FeatureSet, overlay_fbank, pad_shorter
from lhotse.supervision import SupervisionSegment, SupervisionSet
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

    def overlay(self, other: 'Cut', offset_other_by: Seconds = 0.0, snr: Decibels = 0.0) -> 'MixedCut':
        return MixedCut(
            id=str(uuid4()),
            left_cut_id=self.id,
            right_cut_id=other.id,
            offset_right_by=offset_other_by,
            snr=snr
        )

    def append(self, other: 'Cut', snr: float) -> 'MixedCut':
        return self.overlay(other=other, offset_other_by=self.duration, snr=snr)


@dataclass
class MixedCut:
    """
    Represents a Cut that's created from other Cuts via overlay or append operations.
    The actual mixing operations are performed upon loading the features into memory.
    In order to load the features, it needs to access the CutSet object that holds the "ingredient" cuts,
    as it only holds their IDs ("pointers").
    """
    # TODO: it could actually consist of more than two cuts by having a list of "ingredient" ids, offsets and snrs
    id: str
    left_cut_id: str
    right_cut_id: str
    offset_right_by: Seconds
    snr: Decibels

    def with_cut_set(self, cut_set: 'CutSet') -> 'MixedCut':
        # TODO: temporary workaround; think how to design this part better
        #  maybe make MixedCut "own" the Cuts that it consists of (but then the manifests are gonna be really thick)
        #  or just pass the CutSet object in here which is not elegant but maybe better than this
        self._cut_set = cut_set
        return self

    @property
    def supervisions(self) -> List[SupervisionSegment]:
        return self._cut_set.cuts[self.left_cut_id].supervisions + self._cut_set.cuts[self.right_cut_id].supervisions

    @property
    def duration(self) -> Seconds:
        return max(
            self._cut_set.cuts[self.left_cut_id].duration,
            self.offset_right_by + self._cut_set.cuts[self.right_cut_id].duration
        )

    def load_features(self, root_dir: Optional[Pathlike] = None) -> np.ndarray:
        cuts = [self._cut_set.cuts[id_] for id_ in [self.left_cut_id, self.right_cut_id]]
        frame_length, frame_shift = cuts[0].features.frame_length, cuts[0].features.frame_shift
        assert frame_length == cuts[1].features.frame_length
        assert frame_shift == cuts[1].features.frame_shift
        feats = [cut.load_features(root_dir=root_dir) for cut in cuts]
        feats = pad_shorter(*feats)
        overlayed_feats = overlay_fbank(
            feats[0],
            feats[1],
            snr=self.snr,
            offset_right_by=self.offset_right_by,
            frame_length=frame_length,
            frame_shift=frame_shift
        )
        return overlayed_feats


@dataclass
class CutSet:
    """
    CutSet combines features with their corresponding supervisions.
    It may have wider span than the actual supervisions, provided the features for the whole span exist.
    It is the basic building block of PyTorch-style Datasets for speech/audio processing tasks.
    """
    cuts: Dict[str, Union[Cut, MixedCut]]

    @staticmethod
    def from_yaml(path: Pathlike) -> 'CutSet':
        with open(path) as f:
            raw_cuts = yaml.safe_load(f)

        # TODO: refactor into a separate serializer object

        def deserialize_one(cut: Dict[str, Any]):
            cut_type = cut['type']
            del cut['type']

            if cut_type == 'MixedCut':
                return MixedCut(**cut)
            elif cut_type != 'Cut':
                raise ValueError(f"Unexpected cut type during deserialization: '{cut_type}'")

            feature_info = cut['features']
            del cut['features']
            supervision_infos = cut['supervisions']
            del cut['supervisions']
            return Cut(
                **cut,
                features=Features(**feature_info),
                supervisions=[SupervisionSegment(**s) for s in supervision_infos]
            )

        return CutSet(cuts={cut['id']: deserialize_one(cut) for cut in raw_cuts})

    def to_yaml(self, path: Pathlike):
        with open(path, 'w') as f:
            yaml.safe_dump([{**asdict(cut), 'type': type(cut).__name__} for cut in self], stream=f)

    def __len__(self) -> int:
        return len(self.cuts)

    def __iter__(self) -> Iterable[Cut]:
        return iter(self.cuts.values())

    def __add__(self, other: 'CutSet') -> 'CutSet':
        return CutSet(cuts={**self.cuts, **other.cuts})


def make_trivial_cut_set(supervision_set: SupervisionSet, feature_set: FeatureSet) -> CutSet:
    cuts = (
        Cut(
            id=str(uuid4()),
            channel=supervision.channel_id,
            start=supervision.start,
            duration=supervision.duration,
            features=feature_set.find(
                recording_id=supervision.recording_id,
                channel_id=supervision.channel_id,
                start=supervision.start,
                duration=supervision.duration,
            ),
            supervisions=[supervision]
        )
        for idx, supervision in enumerate(supervision_set)
    )
    return CutSet(cuts={cut.id: cut for cut in cuts})
