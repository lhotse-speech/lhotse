from pathlib import Path
from math import isclose
from typing import Dict, Optional

import numpy as np
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

    In the future, will be extended by graph supervisions.
    """

    def __init__(
            self,
            cuts: CutSet,
            duration: Optional[Seconds] = 5.0,
            root_dir: Optional[Pathlike] = None
    ):
        super().__init__()
        self.cuts = cuts.cut_into_windows(duration, keep_excessive_supervisions=True).cuts
        self.root_dir = Path(root_dir) if root_dir else None
        self.cut_ids = list(self.cuts.keys())

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        cut_id = self.cut_ids[idx]
        cut = self.cuts[cut_id]

        features = torch.from_numpy(cut.load_features(root_dir=self.root_dir))
        assert features.shape[0] == cut.num_frames
        assert isclose(cut.num_frames * cut.frame_shift, cut.duration)

        is_voice = np.zeros(cut.num_frames)
        for subversion in cut.supervisions:
            st = round(subversion.start / cut.frame_shift) if subversion.start > 0 else 0
            et = round(subversion.end / cut.frame_shift) if subversion.end < cut.duration else cut.num_frames
            is_voice[st:et] = 1

        return {
            'features': features,
            'is_voice': torch.from_numpy(is_voice)
        }

    def __len__(self) -> int:
        return len(self.cut_ids)
