from collections import defaultdict

import math

import random

from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import torch
from typing_extensions import Protocol, runtime_checkable

from lhotse import CutSet, SupervisionSegment, warnings
from lhotse.cut import AnyCut, MixedCut
from lhotse.utils import Seconds, compute_num_frames


class SupervisionField(Protocol):
    def __call__(self, supervision: SupervisionSegment, cut: AnyCut) -> Dict[str, Any]: ...


@runtime_checkable
class RequiresCustomCollation(Protocol):
    def collate(self, items: Sequence) -> Any:
        pass


class Text:
    def __call__(self, supervision: SupervisionSegment, cut: AnyCut) -> Dict[str, Any]:
        return {'text': supervision.text}


class Characters:
    def __init__(self, cuts):
        self.pad = '<PAD>'
        self.id2sym = sorted({char for cut in cuts for s in cut.supervisions for char in s.text}) + [self.pad]
        self.sym2id = {v: k for k, v in enumerate(self.id2sym)}

    def __call__(self, supervision, cut):
        return {'chars': [self.sym2id[c] for c in supervision.text]}

    def collate(self, items: List[int]):
        items = sorted(items, key=len, reverse=True)
        for idx in range(1, len(items)):
            items[idx].extend([self.sym2id[self.pad]] * (len(items[0]) - len(items[idx])))
        return torch.tensor(items, dtype=torch.int32)


class Speaker:
    def __init__(self, cuts: CutSet):
        self.id2spk = sorted(cuts.speakers)
        self.spk2id = {v: k for k, v in enumerate(self.id2spk)}

    def __call__(self, supervision: SupervisionSegment, cut: AnyCut) -> Dict[str, Any]:
        return {'speaker': self.spk2id[supervision.speaker]}


class VoiceActivity:
    def __init__(self, use_features: bool = True, use_audio: bool = False):
        if not use_audio and not use_features:
            raise ValueError("VoiceActivity requested but both use_audio and use_features set to False.")
        self.use_features = use_features
        self.use_audio = use_audio

    def __call__(self, supervision: SupervisionSegment, cut: AnyCut) -> Dict[str, Any]:
        output = {}
        if self.use_audio:
            output['audio_is_voiced'] = cut.supervisions_audio_mask()
        if self.use_features:
            output['frame_is_voiced'] = cut.supervisions_feature_mask()
        return output


class CutCollater:
    def collate(self, cuts: Sequence[AnyCut]) -> Sequence[AnyCut]:
        return cuts


class SpeechDataset(torch.utils.data.IterableDataset):
    """
    The PyTorch Dataset for the speech recognition task using K2 library.

    This dataset internally batches and collates the Cuts and should be used with
    PyTorch DataLoader with argument batch_size=None to work properly.
    The batch size is determined automatically to satisfy the constraints of ``max_frames``
    and ``max_cuts``.

    This dataset will automatically partition itself when used with a multiprocessing DataLoader
    (i.e. the same cut will not appear twice in the same epoch).

    By default, we "pack" the batches to minimize the amount of padding - we achieve that by
    concatenating the cuts' feature matrices with a small amount of silence (padding) in between.

    Each item in this dataset is a dict of:

    .. code-block::

        {
            'features': float tensor of shape (B, T, F)
            'supervisions': [
                {
                    'cut_id': List[str] of len S
                    'sequence_idx': Tensor[int] of shape (S,)
                    'text': List[str] of len S
                    'start_frame': Tensor[int] of shape (S,)
                    'num_frames': Tensor[int] of shape (S,)
                }
            ]
        }

    Dimension symbols legend:
    * ``B`` - batch size (number of Cuts)
    * ``S`` - number of supervision segments (greater or equal to B, as each Cut may have multiple supervisions)
    * ``T`` - number of frames of the longest Cut
    * ``F`` - number of features

    The 'sequence_idx' field is the index of the Cut used to create the example in the Dataset.
    """

    def __init__(
            self,
            cuts: CutSet,
            fields: Optional[Sequence[SupervisionField]] = None,
            use_features: bool = True,
            use_audio: bool = False,
            multi_channel: bool = False,
            return_cuts: bool = False,
            max_frames: int = 26000,
            max_cuts: Optional[int] = None,
            shuffle: bool = False,
            concat_cuts: bool = True,
            concat_cuts_gap: Seconds = 1.0,
            concat_cuts_duration_factor: float = 1
    ):
        """
        K2 ASR IterableDataset constructor.

        :param cuts: the ``CutSet`` to sample data from.
        :param max_frames: The maximum number of feature frames that we're going to put in a single batch.
            The padding frames do not contribute to that limit, since we pack the batch by default to minimze
            the amount of padding.
        :param max_cuts: The maximum number of cuts sampled to form a mini-batch.
            By default, this constraint is off.
        :param shuffle: When ``True``, the cuts will be shuffled at the start of iteration.
            Convenient when mini-batch loop is inside an outer epoch-level loop, e.g.:
            `for epoch in range(10): for batch in dataset: ...` as every epoch will see a
            different cuts order.
        :param concat_cuts: When ``True``, we will concatenate the cuts to minimize the total amount of padding;
            e.g. instead of creating a batch with 40 examples, we will merge some of the examples together
            adding some silence between them to avoid a large number of padding frames that waste the computation.
            Enabled by default.
        :param concat_cuts_gap: The duration of silence in seconds that is inserted between the cuts;
            it's goal is to let the model "know" that there are separate utterances in a single example.
        :param concat_cuts_duration_factor: Determines the maximum duration of the concatenated cuts;
            by default it's 1, setting the limit at the duration of the longest cut in the batch.
        """
        super().__init__()
        # Initialize the fields
        self.cuts = cuts
        self.shuffle = shuffle
        self.max_frames = max_frames
        self.max_cuts = max_cuts
        self.concat_cuts = concat_cuts
        self.concat_cuts_gap = concat_cuts_gap
        self.concat_cuts_duration_factor = concat_cuts_duration_factor
        # Representations / Supervisions configuration
        self.fields = fields
        self.use_features = use_features
        self.use_audio = use_audio
        self.multi_channel = multi_channel
        self.return_cuts = return_cuts
        # Set-up the mutable state for new epoch initialization in __iter__
        self.cut_ids = list(self.cuts.ids)
        self.current_idx = None
        # Set-up the pseudo-immutable state for multiprocessing DataLoader compatibility
        self.partition_start = 0
        self.partition_end = len(self.cut_ids)
        self._validate()

    def __iter__(self):
        """
        Prepare the dataset for iterating over a new epoch. Will shuffle the data if requested.

        This method takes care of partitioning for multiprocess data loading, so that the
        dataset won't return data duplicates within a single epoch (for more details, see:
        https://pytorch.org/docs/stable/data.html at "Multi-process data loading").
        """
        # noinspection PyUnresolvedReferences
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # No multiprocessing involved - iterate full the CutSet.
            self.partition_start = 0
            self.partition_end = len(self.cut_ids)
        else:
            # We are in a worker process - need to select a partition to process.
            start, end = 0, len(self.cut_ids)
            per_worker = int(math.ceil((end - start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            self.partition_start = start + worker_id * per_worker
            self.partition_end = min(self.partition_start + per_worker, end)

        # Re-set the mutable state.
        # Iterating over this dataset is equivalent to starting a new epoch.
        if self.shuffle:
            # If we're in multiprocessing mode, we should shuffle only the within the partition
            # given to this worker, otherwise we might use data samples that are not intended
            # for this worker.
            partition_cut_ids = self.cut_ids[self.partition_start: self.partition_end]
            random.shuffle(partition_cut_ids)
            self.cut_ids[self.partition_start: self.partition_end] = partition_cut_ids
        self.current_idx = self.partition_start
        return self

    def __next__(self) -> Dict[str, Union[torch.Tensor, List[str]]]:
        """
        Return a new batch, with the batch size automatically determined using the contraints
        of max_frames and max_cuts.
        """
        from torch.utils.data._utils.collate import default_collate

        # Collect the cuts that will form a batch, satisfying the criteria of max_cuts and max_frames.
        # The returned object is a CutSet that we can keep on modifying (e.g. padding, mixing, etc.)
        cuts: CutSet = self._collect_batch()

        # For now, we'll just pad it with low energy values to match the longest Cut's
        # duration in the batch. We might want to do something more interesting here
        # later on - padding/mixing with noises, etc.
        # TODO: multiple ways to pad the cuts
        cuts = cuts.sort_by_duration().pad()

        output = {}

        if self.use_features:
            if self.multi_channel:
                output['features'] = _collate_multi_channel_features(cuts)
            else:  # single channel
                # Get a tensor with batched feature matrices, shape (B, T, F)
                output['features'] = _collate_features(cuts)

        if self.use_audio:
            if self.multi_channel:
                output['audio'] = _collate_multi_channel_audio(cuts)
            else:  # single channel
                output['audio'] = _collate_audio(cuts)

        if self.fields:
            supervisions = []
            custom_collated = defaultdict(list)
            collaters: Dict[str, RequiresCustomCollation] = {}
            for sequence_idx, cut in enumerate(cuts):
                for supervision in cut.supervisions:
                    info = {
                        'cut_id': cut.id,
                        'sequence_idx': sequence_idx
                    }
                    if self.return_cuts:
                        custom_collated['cut'].append(cut)
                        collaters['cut'] = CutCollater()
                    if self.use_features:
                        info['start_frame'] = compute_num_frames(supervision.start, cut.frame_shift),
                        info['num_frames'] = _asserted_num_frames(
                            output['features'].shape[2 if self.multi_channel else 1],
                            supervision.start,
                            supervision.duration,
                            cut.frame_shift
                        )
                    if self.use_audio:
                        info['start_sample'] = round(supervision.start * cut.sampling_rate)
                        info['num_samples'] = round(supervision.duration * cut.sampling_rate)
                    for supervision_type in self.fields:
                        entry = supervision_type(supervision, cut)
                        if isinstance(supervision_type, RequiresCustomCollation):
                            for key, value in entry.items():
                                custom_collated[key].append(value)
                                if key not in collaters:
                                    collaters[key] = supervision_type
                        else:
                            info.update(entry)
                    supervisions.append(info)
            output['supervisions'] = default_collate(supervisions)
            output['supervisions'].update(
                {key: collaters[key].collate(values) for key, values in custom_collated.items()}
            )

        return output

    def _collect_batch(self) -> CutSet:
        """
        Return a sub-CutSet that represents a full batch.
        This is quick, as it does not perform any I/O in the process.
        """
        # Keep iterating the underlying CutSet as long as we hit or exceed the constraints
        # provided by user (the max number of frames or max number of cuts).
        # Note: no actual data is loaded into memory yet because the manifests contain all the metadata
        # required to do this operation.
        num_frames = 0
        cuts = []
        while True:
            # Check that we have not reached the end of the dataset.
            if self.current_idx < self.partition_end:
                # We didn't - grab the next cut
                next_cut_id = self.cut_ids[self.current_idx]
            else:
                if cuts:
                    # We did and we have a partial batch - return it.
                    return CutSet.from_cuts(cuts)
                else:
                    # We did and there is nothing more to return - signal the iteration code to stop.
                    raise StopIteration()
            next_cut = self.cuts[next_cut_id]
            next_num_frames = num_frames + next_cut.num_frames
            next_num_cuts = len(cuts) + 1
            # Did we exceed the max_frames and max_cuts constraints?
            if next_num_frames <= self.max_frames and (self.max_cuts is None or next_num_cuts <= self.max_cuts):
                # No - add the next cut to the batch, and keep trying.
                num_frames = next_num_frames
                cuts.append(next_cut)
                self.current_idx += 1
            else:
                # Yes. Do we have at least one cut in the batch?
                if cuts:
                    # Yes. Return it.
                    break
                else:
                    # No. We'll warn the user that the constrains might be too tight,
                    # and return the cut anyway.
                    warnings.warn("The first cut drawn in batch collection violates the max_frames or max_cuts "
                                  "constraints - we'll return it anyway. Consider increasing max_frames/max_cuts.")
                    cuts.append(next_cut)
                    self.current_idx += 1
        if self.concat_cuts:
            cuts = concat_cuts(
                cuts,
                gap=self.concat_cuts_gap,
                max_duration=self.concat_cuts_duration_factor * cuts[0].duration
            )
        return CutSet.from_cuts(cuts)

    def _validate(self) -> None:
        for cut in self.cuts:
            for supervision in cut.supervisions:
                assert cut.start <= supervision.start <= supervision.end <= cut.end, \
                    f"Cutting in the middle of a supervision is currently not supported for the ASR task. " \
                    f"Cut ID violating the pre-condition: '{cut.id}'"
        assert self.max_frames > 0
        assert self.max_cuts is None or self.max_cuts > 0


def _collate_features(cuts: CutSet) -> torch.Tensor:
    assert all(cut.has_features for cut in cuts)
    first_cut = next(iter(cuts))
    features = torch.empty(len(cuts), first_cut.num_frames, first_cut.num_features)
    for idx, cut in enumerate(cuts):
        features[idx] = torch.from_numpy(cut.load_features())
    return features


def _collate_audio(cuts: CutSet) -> torch.Tensor:
    assert all(cut.has_recording for cut in cuts)
    first_cut = next(iter(cuts))
    audio = torch.empty(len(cuts), first_cut.num_samples)
    for idx, cut in enumerate(cuts):
        audio[idx] = torch.from_numpy(cut.load_audio()[0])
    return audio


def _collate_multi_channel_features(cuts: CutSet) -> torch.Tensor:
    assert all(cut.has_features for cut in cuts)
    assert all(isinstance(cut, MixedCut) for cut in cuts)
    # Output tensor shape: (B, C, T, F) -> (batch_size, num_channels, num_frames, num_features)
    first_cut = next(iter(cuts))
    # TODO: make MixedCut more friendly to use with multi channel audio;
    #  discount PaddingCuts in "tracks" when specifying the number of channels
    features = torch.empty(len(cuts), len(first_cut.tracks), first_cut.num_frames, first_cut.num_features)
    for idx, cut in enumerate(cuts):
        features[idx] = torch.from_numpy(cut.load_features(mixed=False))
    return features


def _collate_multi_channel_audio(cuts: CutSet) -> torch.Tensor:
    assert all(cut.has_recording for cut in cuts)
    assert all(isinstance(cut, MixedCut) for cut in cuts)
    first_cut = next(iter(cuts))
    audio = torch.empty(len(cuts), len(first_cut.tracks), first_cut.num_samples)
    for idx, cut in enumerate(cuts):
        audio[idx] = torch.from_numpy(cut.load_audio())
    return audio


def _asserted_num_frames(total_num_frames: int, start: Seconds, duration: Seconds, frame_shift: Seconds) -> int:
    """
    This closure with compute the num_frames, correct off-by-one errors in edge cases,
    and assert that the supervision does not exceed the feature matrix temporal dimension.
    """
    offset = compute_num_frames(start, frame_shift=frame_shift)
    num_frames = compute_num_frames(duration, frame_shift=frame_shift)
    diff = total_num_frames - (offset + num_frames)
    # Note: we tolerate off-by-ones because some mixed cuts could have one frame more
    # than their duration suggests (we will try to change this eventually).
    if diff == -1:
        num_frames -= 1
    assert offset + num_frames <= total_num_frames, \
        f"Unexpected num_frames ({offset + num_frames}) exceeding features time dimension for a supervision " \
        f"({total_num_frames}) when constructing a batch; please report this in Lhotse's GitHub issues, " \
        "ideally providing the Cut data that triggered this."
    return num_frames


def concat_cuts(
        cuts: List[AnyCut],
        gap: Seconds = 1.0,
        max_duration: Optional[Seconds] = None
) -> List[AnyCut]:
    """
    We're going to concatenate the cuts to minimize the amount of total padding frames used.
    This is actually solving a knapsack problem.
    In this initial implementation we're using a greedy approach:
    going from the back (i.e. the shortest cuts) we'll try to concat them to the longest cut
    that still has some "space" at the end.

    :param cuts: a list of cuts to pack.
    :param gap: the duration of silence inserted between concatenated cuts.
    :param max_duration: the maximum duration for the concatenated cuts
        (by default set to the duration of the first cut).
    :return a list of packed cuts.
    """
    if len(cuts) <= 1:
        return cuts
    cuts = sorted(cuts, key=lambda c: c.duration, reverse=True)
    max_duration = cuts[0].duration if max_duration is None else max_duration
    current_idx = 1
    while True:
        can_fit = False
        shortest = cuts[-1]
        for idx in range(current_idx, len(cuts) - 1):
            cut = cuts[current_idx]
            can_fit = cut.duration + gap + shortest.duration <= max_duration
            if can_fit:
                cuts[current_idx] = cut.pad(cut.duration + gap).append(shortest)
                cuts = cuts[:-1]
                break
            current_idx += 1
        if not can_fit:
            break
    return cuts
