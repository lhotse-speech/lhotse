from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import Dataset

from lhotse.cut import CutSet
from lhotse.utils import Pathlike

EPS = 1e-8


class SourceSeparationDataset(Dataset):
    """
    A PyTorch Dataset for the source separation task. It's created from two CutSets - one provides the audio
    cuts for the sources, and the other one the audio cuts for the signal mix. When queried for data samples,
    it returns a dict of {'sources': tensor, 'mixture': tensor}.
    """

    def __init__(
            self,
            sources_set: CutSet,
            mixtures_set: CutSet,
            root_dir: Optional[Pathlike] = None
    ):
        super().__init__()
        self.sources_set = sources_set
        self.mixtures_set = mixtures_set
        self.cut_ids = list(self.mixtures_set.cuts.keys())
        self.root_dir = Path(root_dir) if root_dir else None
        assert set(self.cut_ids) == set(self.mixtures_set.cuts.keys())

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        cut_id = self.cut_ids[idx]
        mixture_cut = self.mixtures_set.cuts[cut_id].with_cut_set(self.sources_set)
        source_cuts = [self.sources_set.cuts[id_] for id_ in [mixture_cut.left_cut_id, mixture_cut.right_cut_id]]
        mixture = torch.from_numpy(mixture_cut.load_features(root_dir=self.root_dir))
        sources = torch.stack(
            [torch.from_numpy(source_cut.load_features(root_dir=self.root_dir)) for source_cut in source_cuts],
            dim=0
        )

        # Compute the masks given the source features
        real_mask = sources / (sources.sum(1, keepdim=True) + EPS)
        # Get the src idx having the maximum energy
        binary_mask = real_mask.argmax(1)

        return {
            'sources': sources,
            'mixture': mixture,
            'real_mask': real_mask,
            'binary_mask': binary_mask
        }

    def __len__(self):
        return len(self.cut_ids)
