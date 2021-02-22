from typing import Dict, Iterable

import torch

from lhotse import validate
from lhotse.cut import CutSet
from lhotse.dataset.collation import collate_audio, collate_features


class SpeechSynthesisDataset(torch.utils.data.Dataset):
    """
    The PyTorch Dataset for the speech synthesis task.
    Each item in this dataset is a dict of:

    .. code-block::

        {
            'audio': (B x NumSamples) tensor
            'features': (B x NumFrames x NumFeatures) tensor
            'tokens': list of lists of characters (str)
        }
    """

    def __init__(self, cuts: CutSet) -> None:
        super().__init__()
        validate(cuts)
        self.cuts = cuts

        # generate tokens from text
        self.cut_id_to_token = {}
        self.tokens = set()
        for cut in cuts:
            assert len(cut.supervisions) == 1, 'Only the Cuts with single supervision are supported.'
            characters = list(cut.supervisions[0].text)
            self.tokens.update(set(characters))
            self.cut_id_to_token[cut.id] = characters
        self.tokens = sorted(list(self.tokens))

    def __getitem__(self, cut_ids: Iterable[str]) -> Dict[str, torch.Tensor]:
        cuts = self.cuts.subset(cut_ids=cut_ids)

        features = collate_features(cuts)
        audio = collate_audio(cuts)
        return {
            'audio': audio,
            'features': features,
            # TODO: consider extending the dataset to create a collated tensor of integer token IDs
            'tokens': [self.cut_id_to_token[cut.id] for cut in cuts]
        }

    def __len__(self) -> int:
        return len(self.cut_ids)
