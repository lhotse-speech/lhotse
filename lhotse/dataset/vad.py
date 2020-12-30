from lhotse.dataset import SpeechDataset, fields


class VadDataset(SpeechDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            signal_fields=[fields.Feats()],
            supervision_fields=[fields.VoiceActivity()],
            **kwargs
        )
