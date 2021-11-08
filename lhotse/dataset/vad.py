from typing import Callable, Dict, Sequence

import torch

from lhotse import validate
from lhotse.cut import CutSet
from lhotse.dataset.input_strategies import BatchIO, PrecomputedFeatures
from lhotse.utils import ifnone


class VadDataset(torch.utils.data.Dataset):
    """
    The PyTorch Dataset for the voice activity detection task.
    Each item in this dataset is a dict of:

    .. code-block::

        {
            'inputs': (B x T x F) tensor
            'input_lens': (B,) tensor
            'is_voice': (T x 1) tensor
            'cut': List[Cut]
        }
    """

    def __init__(
        self,
        input_strategy: BatchIO = PrecomputedFeatures(),
        cut_transforms: Sequence[Callable[[CutSet], CutSet]] = None,
        input_transforms: Sequence[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.input_strategy = input_strategy
        self.cut_transforms = ifnone(cut_transforms, [])
        self.input_transforms = ifnone(input_transforms, [])

    def __getitem__(self, cuts: CutSet) -> Dict[str, torch.Tensor]:
        validate(cuts)
        cuts = cuts.sort_by_duration()
        for tfnm in self.cut_transforms:
            cuts = tfnm(cuts)
        inputs, input_lens = self.input_strategy(cuts)
        for tfnm in self.input_transforms:
            inputs = tfnm(inputs)
        return {
            "inputs": inputs,
            "input_lens": input_lens,
            "is_voice": self.input_strategy.supervision_masks(cuts),
            "cut": cuts,
        }
