from typing import Optional

import torch
from torch.utils.data import Dataset

from lhotse.cut import CutSet
from lhotse.utils import Pathlike


class UnsupervisedDataset(Dataset):
    """
    Dataset that contains no supervision - it only provides the features extracted from recordings.
    The returned features are a :class:`torch.Tensor` of shape ``(T x F)``, where T is the number of frames,
    and F is the feature dimension.
    """

    def __init__(self, cuts: CutSet, root_dir: Optional[Pathlike] = None):
        super().__init__()
        self.cuts = cuts
        self.cut_ids = list(self.cuts.cuts.keys())
        self.root_dir = root_dir

    def __getitem__(self, item: int) -> torch.Tensor:
        cut = self.cuts[self.cut_ids[item]]
        feats = cut.load_features(root_dir=self.root_dir)
        return torch.from_numpy(feats)

    def __len__(self):
        return len(self.cuts)
