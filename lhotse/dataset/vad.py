from math import isclose
from typing import Dict

import torch
from torch.utils.data import Dataset

from lhotse.cut import CutSet

EPS = 1e-8


class VadDataset(Dataset):
    """
    The PyTorch Dataset for the voice activity detection task.
    Each item in this dataset is a dict of:

    .. code-block::

        {
            'features': (T x F) tensor
            'is_voice': (T x 1) tensor
        }
    """

    def __init__(
            self,
            cuts: CutSet,
    ):
        super().__init__()
        self.cuts = cuts
        self.cut_ids = list(cuts.ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        cut_id = self.cut_ids[idx]
        cut = self.cuts[cut_id]

        features = torch.from_numpy(cut.load_features())
        assert features.shape[0] == cut.num_frames
        assert isclose(cut.num_frames * cut.frame_shift, cut.duration)

        is_voice = torch.from_numpy(cut.supervisions_feature_mask())

        return {
            'features': features,
            'is_voice': is_voice
        }

    def __len__(self) -> int:
        return len(self.cut_ids)
