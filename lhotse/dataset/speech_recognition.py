import logging
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import Dataset

from lhotse.cut import CutSet
from lhotse.utils import Pathlike

EPS = 1e-8


class SpeechRecognitionDataset(Dataset):
    """
    The PyTorch Dataset for the speech recognition task.
    Returns a dict of:

    .. code-block::

        {
            'features': (T x F) tensor
            'text': string
        }

    In the future, will be extended by graph supervisions.
    """

    def __init__(
            self,
            cuts: CutSet,
            root_dir: Optional[Pathlike] = None
    ):
        super().__init__()
        self.cuts = cuts.trim_to_supervisions()
        self.root_dir = Path(root_dir) if root_dir else None
        self.cut_ids = list(self.cuts.cuts.keys())

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        cut_id = self.cut_ids[idx]
        cut = self.cuts[cut_id]

        features = torch.from_numpy(cut.load_features(root_dir=self.root_dir))

        # There should be only one supervision because we have had trim_to_supervisions() processed
        assert len(cut.supervisions) == 1, "SpeechRecognitionDataset does not support overlapping supervisions yet."

        return {
            'features': features,
            'text': cut.supervisions[0].text
        }

    def __len__(self) -> int:
        return len(self.cut_ids)
