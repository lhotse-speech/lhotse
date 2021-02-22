from typing import Dict, Iterable

import torch

from lhotse import validate
from lhotse.cut import CutSet
from lhotse.dataset.collation import collate_features, collate_vectors


class VadDataset(torch.utils.data.Dataset):
    """
    The PyTorch Dataset for the voice activity detection task.
    Each item in this dataset is a dict of:

    .. code-block::

        {
            'features': (T x F) tensor
            'is_voice': (T x 1) tensor
            'cut': List[Cut]
        }
    """

    def __init__(self, cuts: CutSet) -> None:
        super().__init__()
        validate(cuts)
        self.cuts = cuts

    def __getitem__(self, cut_ids: Iterable[str]) -> Dict[str, torch.Tensor]:
        cuts = self.cuts.subset(cut_ids=cut_ids).sort_by_duration()
        return {
            'features': collate_features(cuts),
            'is_voice': collate_vectors(c.supervisions_feature_mask() for c in cuts),
            'cut': cuts
        }

    def __len__(self) -> int:
        return len(self.cuts)
