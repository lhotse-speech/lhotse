from pathlib import Path
from math import isclose
from typing import Dict, Optional

import torch
from torch.utils.data import Dataset

from lhotse.cut import CutSet
from lhotse.utils import Pathlike, Seconds

EPS = 1e-8


class VadDataset(Dataset):
    """
    The PyTorch Dataset for the voice activity detection task.
    Returns a dict of:

    .. code-block::

        {
            'features': (T x F) tensor
            'is_voice': (T x 1) tensor
        }
    """

    def __init__(
            self,
            cuts: CutSet,
            root_dir: Optional[Pathlike] = None
    ):
        super().__init__()
        self.cuts = cuts
        self.root_dir = Path(root_dir) if root_dir else None
        self.cut_ids = list(cuts.ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        cut_id = self.cut_ids[idx]
        cut = self.cuts[cut_id]

        features = torch.from_numpy(cut.load_features(root_dir=self.root_dir))
        assert features.shape[0] == cut.num_frames
        assert isclose(cut.num_frames * cut.frame_shift, cut.duration)

        is_voice = torch.zeros(cut.num_frames)
        for supervision in cut.supervisions:
            st = round(supervision.start / cut.frame_shift) if supervision.start > 0 else 0
            et = round(supervision.end / cut.frame_shift) if supervision.end < cut.duration else cut.num_frames
            is_voice[st:et] = 1

        return {
            'features': features,
            'is_voice': is_voice
        }

    def __len__(self) -> int:
        return len(self.cut_ids)
