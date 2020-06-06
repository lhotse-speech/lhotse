from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import Dataset

from lhotse.cut import CutSet
from lhotse.utils import Pathlike


class SourceSeparationDataset(Dataset):
    def __init__(self, sources_set: CutSet, mixtures_set: CutSet, root_dir: Pathlike):
        super().__init__()
        self.sources_set = sources_set
        self.mixtures_set = mixtures_set
        self.cut_ids = list(self.mixtures_set.cuts.keys())
        self.root_dir = Path(root_dir)
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
        return {
            'sources': sources,
            'mixture': mixture
        }

    def __len__(self):
        return len(self.cut_ids)
