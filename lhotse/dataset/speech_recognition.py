import math
import random
from typing import Dict, List, Optional, Sequence, Union

import torch
from torch.utils.data.dataloader import DataLoader

from lhotse.cut import CutSet

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
    This dataset will automatically partition itself when used with a multiprocessing DataLoader.

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
            shuffle: bool = False
    ):
        """
        K2 ASR IterableDataset constructor.

        :param cuts: the ``CutSet`` to sample data from.
        :param max_frames: The maximum number of feature frames that we're going to put in a single batch.
            This number includes the padding frames.
        :param max_cuts: The maximum number of cuts sampled to form a mini-batch.
            By default, this constraint is off.
        :param shuffle: When ``True``, the cuts will be shuffled at the start of iteration.
            Convenient when mini-batch loop is inside an outer epoch-level loop, e.g.:
            `for epoch in range(10): for batch in dataset: ...` as every epoch will see a
            different cuts order.
        """
        super().__init__()
        # Initialize the fields
        self.cuts = cuts
        self.shuffle = shuffle
        self.max_frames = max_frames
        self.max_cuts = max_cuts
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

        return {
            'features': features,
            'supervisions': default_collate([
                {
                    'cut_id': cut.id,
                    'sequence_idx': sequence_idx,
                    'text': supervision.text,
                    'start_frame': round(supervision.start / cut.frame_shift),
                    'num_frames': round(supervision.duration / cut.frame_shift),
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
            if next_num_frames <= self.max_frames and (self.max_cuts is None or next_num_cuts <= self.max_cuts):
                num_frames = next_num_frames
                cuts.append(next_cut)
                self.current_idx += 1
            else:
                # Note: in a more elaborate collection scheme, we would try to find some small cut
                # to "squeeze in" to satisfy the constraints as best as we can; we do no such thing
                # in this initial implementation.
                break
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
