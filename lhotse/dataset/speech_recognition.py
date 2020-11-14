import math
import random
import warnings
from typing import Dict, List, Optional, Sequence, Union

import torch
from torch.utils.data.dataloader import DataLoader

from lhotse.cut import AnyCut, CutSet
from lhotse.utils import Seconds, compute_num_frames

EPS = 1e-8


class SpeechRecognitionDataset(torch.utils.data.Dataset):
    """
    The PyTorch Dataset for the speech recognition task.
    Each item in this dataset is a dict of:

    .. code-block::

        {
            'features': (T x F) tensor,
            'text': string,
            'supervisions_mask': (T) tensor
        }

    The ``supervisions_mask`` field is a mask that specifies which frames are covered by a supervision
    by assigning a value of 1 (in this case: segments with transcribed speech contents),
    and which are not by asigning a value of 0 (in this case: padding, contextual noise,
    or in general the acoustic context without transcription).

    In the future, will be extended by graph supervisions.
    """

    def __init__(self, cuts: CutSet):
        super().__init__()
        self.cuts = cuts
        self.cut_ids = list(self.cuts.ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        cut_id = self.cut_ids[idx]
        cut = self.cuts[cut_id]

        features = torch.from_numpy(cut.load_features())
        mask = torch.from_numpy(cut.supervisions_feature_mask())

        # There should be only one supervision because we expect that trim_to_supervisions() was called,
        # or the dataset was created from pre-segment recordings
        assert len(cut.supervisions) == 1, "SpeechRecognitionDataset does not support multiple supervisions yet. " \
                                           "Use CutSet.trim_to_supervisions() to cut long recordings into short " \
                                           "supervisions segment, and follow up with either .pad(), " \
                                           ".truncate(), and possibly .filter() to make sure that all cuts " \
                                           "have a uniform duration."

        return {
            'features': features,
            'text': cut.supervisions[0].text,
            'supervisions_mask': mask
        }

    def __len__(self) -> int:
        return len(self.cut_ids)


class K2SpeechRecognitionIterableDataset(torch.utils.data.IterableDataset):
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
            max_frames: int = 26000,
            max_cuts: Optional[int] = None,
            shuffle: bool = False,
            concat_cuts: bool = True,
            concat_cuts_gap: Seconds = 1.0,
            concat_cuts_duration_factor: float = 2
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
            by default it's twice the duration of the longest cut in the batch.
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
        cuts = cuts.sort_by_duration().pad()

        # Get a tensor with batched feature matrices, shape (B, T, F)
        features = _collate_features(cuts)

        def asserted_num_frames(start: Seconds, duration: Seconds, frame_shift: Seconds) -> int:
            """
            This closure with compute the num_frames, correct off-by-one errors in edge cases,
            and assert that the supervision does not exceed the feature matrix temporal dimension.
            """
            offset = compute_num_frames(start, frame_shift=frame_shift)
            num_frames = compute_num_frames(duration, frame_shift=frame_shift)
            diff = features.shape[1] - (offset + num_frames)
            # Note: we tolerate off-by-ones because some mixed cuts could have one frame more
            # than their duration suggests (we will try to change this eventually).
            if diff == -1:
                num_frames -= 1
            assert offset + num_frames <= features.shape[1], \
                f"Unexpected num_frames ({offset + num_frames}) exceeding features time dimension for a supervision " \
                f"({features.shape[1]}) when constructing a batch; please report this in Lhotse's GitHub issues, " \
                "ideally providing the Cut data that triggered this."
            return num_frames

        return {
            'features': features,
            'supervisions': default_collate([
                {
                    'cut_id': cut.id,
                    'sequence_idx': sequence_idx,
                    'text': supervision.text,
                    'start_frame': compute_num_frames(supervision.start, cut.frame_shift),
                    'num_frames': asserted_num_frames(supervision.start, supervision.duration, cut.frame_shift),
                }
                for sequence_idx, cut in enumerate(cuts)
                for supervision in cut.supervisions
            ])
        }

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
    first_cut = next(iter(cuts))
    features = torch.empty(len(cuts), first_cut.num_frames, first_cut.num_features)
    for idx, cut in enumerate(cuts):
        features[idx] = torch.from_numpy(cut.load_features())
    return features


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


class K2SpeechRecognitionDataset(torch.utils.data.Dataset):
    """
    The PyTorch Dataset for the speech recognition task using K2 library.
    Each item in this dataset is a dict of:

    .. code-block::

        {
            'features': (T x F) tensor,
            'supervisions': List[Dict] -> [
                {
                    'sequence_idx': int
                    'text': string,
                    'start_frame': int,
                    'num_frames': int
                } (multiplied N times, for each of the N supervisions present in the Cut)
            ]
        }

    The 'sequence_idx' field is the index of the Cut used to create the example in the Dataset.
    It is mapped to the batch index later in the DataLoader.
    """

    def __init__(self, cuts: CutSet):
        super().__init__()
        self.cuts = cuts
        self.cut_ids = list(self.cuts.ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        cut_id = self.cut_ids[idx]
        cut = self.cuts[cut_id]

        features = torch.from_numpy(cut.load_features())

        return {
            'features': features,
            'supervisions': [
                {
                    'sequence_idx': idx,
                    'text': sup.text,
                    'start_frame': round(sup.start / cut.frame_shift),
                    'num_frames': round(sup.duration / cut.frame_shift),
                }
                # CutSet's supervisions can exceed the cut, when the cut starts/ends in the middle
                # of a supervision (they would have relative times e.g. -2 seconds start, meaning
                # it started 2 seconds before the Cut starts). We use s.trim() to get rid of that
                # property, ensuring the supervision time span does not exceed that of the cut.
                for sup in (s.trim(cut.duration) for s in cut.supervisions)
            ]
        }

    def __len__(self) -> int:
        return len(self.cut_ids)


class K2DataLoader(DataLoader):
    """
    A PyTorch DataLoader that has a custom collate_fn that complements the K2SpeechRecognitionDataset.

    The 'features' tensor is collated in a standard way to return a tensor of shape (B, T, F).

    The 'supervisions' dict contains the same fields as in ``K2SpeechRecognitionDataset``,
    except that each sub-field (like 'start_frame') is a 1D PyTorch tensor with shape (B,).
    The 'text' sub-field is an exception - it's a list of strings with length equal to batch size.

    The 'sequence_idx' sub-field in 'supervisions', which originally points to index of the example
    in the Dataset, is remapped to the index of the corresponding features matrix in the
    collated 'features'.
    Multiple supervisions coming from the same cut will share the same 'sequence_idx'.

    For an example, see ``test/dataset/test_speech_recognition_dataset.py::test_k2_dataloader()``.
    """

    def __init__(self, *args, **kwargs):
        if 'collate_fn' in kwargs:
            raise ValueError('Cannot override collate_fn in K2DataLoader.')
        super().__init__(*args, collate_fn=multi_supervision_collate_fn, **kwargs)


def multi_supervision_collate_fn(batch: Sequence[Dict]) -> Dict:
    """
    Custom collate_fn for K2SpeechRecognitionDataset.

    It merges the items provided by K2SpeechRecognitionDataset into the following structure:

    .. code-block::

        {
            'features': float tensor of shape (B, T, F)
            'supervisions': [
                {
                    'sequence_idx': Tensor[int] of shape (S,)
                    'text': List[str] of len S
                    'start_frame': Tensor[int] of shape (S,)
                    'num_frames': Tensor[int] of shape (S,)
                }
            ]
        }

    Dimension symbols legend:
    * ``B`` - batch size (number of Cuts),
    * ``S`` - number of supervision segments (greater or equal to B, as each Cut may have multiple supervisions),
    * ``T`` - number of frames of the longest Cut
    * ``F`` - number of features
    """
    from torch.utils.data._utils.collate import default_collate

    dataset_idx_to_batch_idx = {
        example['supervisions'][0]['sequence_idx']: batch_idx
        for batch_idx, example in enumerate(batch)
    }

    def update(d: Dict, **kwargs) -> Dict:
        for key, value in kwargs.items():
            d[key] = value
        return d

    supervisions = default_collate([
        update(sup, sequence_idx=dataset_idx_to_batch_idx[sup['sequence_idx']])
        for example in batch
        for sup in example['supervisions']
    ])
    feats = default_collate([example['features'] for example in batch])
    return {
        'features': feats,
        'supervisions': supervisions
    }
