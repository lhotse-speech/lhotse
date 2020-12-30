from lhotse.cut import CutSet
from lhotse.dataset.core import SpeechDataset
from lhotse.dataset import fields


class SpeechSynthesisDataset(SpeechDataset):
    def __init__(self, cuts: CutSet, *args, **kwargs):
        super().__init__(
            cuts,
            *args,
            signal_fields=[fields.Audio(), fields.Feats()],
            supervision_fields=[fields.CharacterIds(cuts)],
            **kwargs
        )
