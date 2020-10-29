from typing import Dict, Sequence

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from lhotse.cut import CutSet

EPS = 1e-8


class SpeechRecognitionDataset(Dataset):
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


class K2SpeechRecognitionDataset(Dataset):
    """
    The PyTorch Dataset for the speech recognition task using K2 library.
    Each item in this dataset is a dict of:

    .. code-block::

        {
            'features': (T x F) tensor,
            'supervisions': List[Dict] -> [
                {
                    'example_idx': int
                    'text': string,
                    'start_frame': int,
                    'end_frame': int
                } (multiplied N times, for each of the N supervisions present in the Cut)
            ]
        }

    The 'example_idx' field is the index of the Cut used to create the example in the Dataset.
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
                    'example_idx': idx,
                    'text': sup.text,
                    'start_frame': round(sup.start / cut.frame_shift),
                    'end_frame': round(sup.end / cut.frame_shift),
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

    The 'supervisions' dict contains each supervision as a separate entry
    (its len() might be more than B dim in 'features').
    The 'example_idx' field which originally points to index of the example in the Dataset
    is remapped to the index of the corresponding features matrix in the collated 'features'.
    Multiple supervisions coming from the same cut will share the same 'example_idx'.
    """

    def __init__(self, *args, **kwargs):
        if 'collate_fn' in kwargs:
            raise ValueError('Cannot override collate_fn in K2DataLoader.')
        super().__init__(*args, collate_fn=multi_supervision_collate_fn, **kwargs)


def multi_supervision_collate_fn(batch: Sequence[Dict]) -> Dict:
    """Custom collate_fn for K2SpeechRecognitionDataset. See its docs for details."""
    from torch.utils.data._utils.collate import default_collate

    dataset_idx_to_batch_idx = {
        example['supervisions'][0]['example_idx']: batch_idx
        for batch_idx, example in enumerate(batch)
    }

    def update(d: Dict, **kwargs) -> Dict:
        for key, value in kwargs.items():
            d[key] = value
        return d

    supervisions = default_collate([
        update(sup, example_idx=dataset_idx_to_batch_idx[sup['example_idx']])
        for example in batch
        for sup in example['supervisions']
    ])
    feats = default_collate([example['features'] for example in batch])
    return {
        'features': feats,
        'supervisions': supervisions
    }
