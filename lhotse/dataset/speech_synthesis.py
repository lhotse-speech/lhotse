import logging
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
        self.id_to_tokens = {}
        self.token_set = set()
        for cut in cuts:
            if len(cut.supervisions) > 1:
                logging.warning(f'Cut with multiple supervisions is not supported for TTS: {cut.recording_id}.')
                continue
            tokens = list(cut.supervisions[0].text)
            self.token_set.update(set(tokens))
            self.id_to_tokens[cut.id] = tokens

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        cut_id = self.cut_ids[idx]
        cut = self.cuts[cut_id]

        features = torch.from_numpy(cut.load_features())
        audio = torch.from_numpy(cut.load_audio())
        assert cut.id in self.id_to_tokens
        return {
            'audio': audio,
            'features': features,
            'tokens': self.id_to_tokens[cut.id]
        }

    def __len__(self) -> int:
        return len(self.cut_ids)

    @property
    def vocabulary(self) -> List[str]:
        return sorted(list(self.token_set))
