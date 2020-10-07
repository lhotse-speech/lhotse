from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from lhotse.cut import CutSet
from lhotse.utils import Pathlike

EPS = 1e-8


class SpeechSynthesisDataset(Dataset):
    """
    The PyTorch Dataset for the speech synthesis task.
    Each item in this dataset is a dict of:

    .. code-block::

        {
            'audio': (1 x NumSamples) tensor
            'features': (NumFrames x NumFeatures) tensor
            'tokens': list of characters
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
        self.cut_ids = list(self.cuts.ids)

        # generate tokens from text
        self.id_to_token = {}
        self.token_set = set()
        for cut in cuts:
            assert len(cut.supervisions) == 1, 'Only the Cuts with single supervision are supported.'
            characters = list(cut.supervisions[0].text)
            self.token_set.update(set(characters))
            self.id_to_token[cut.id] = characters
        self.token_set = sorted(list(self.tokens))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        cut_id = self.cut_ids[idx]
        cut = self.cuts[cut_id]

        features = torch.from_numpy(cut.load_features())
        audio = torch.from_numpy(cut.load_audio())
        assert cut.id in self.id_to_token
        return {
            'audio': audio,
            'features': features,
            'tokens': self.id_to_token[cut.id]
        }

    def __len__(self) -> int:
        return len(self.cut_ids)

    @property
    def tokens(self) -> List[str]:
        return self.token_set
