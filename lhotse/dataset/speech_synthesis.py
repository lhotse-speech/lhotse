from typing import Dict, Iterable, Callable, Union, Sequence

import torch

from lhotse import validate
from lhotse.cut import CutSet
from lhotse.dataset.collation import TokenCollater, collate_features, collate_audio


class SpeechSynthesisDataset(torch.utils.data.Dataset):
    """
    The PyTorch Dataset for the speech synthesis task.
    Each item in this dataset is a dict of:

    .. code-block::

        {
            'audio': (B x NumSamples) float tensor
            'features': (B x NumFrames x NumFeatures) float tensor
            'tokens': (B x NumTokens) int tensor
            'audio_lens': (B, ) int tensor
            'features_lens': (B, ) int tensor
            'tokens_lens': (B, ) int tensor
        }
    """

    def __init__(
        self,
        cuts: CutSet,
        feature_transforms: Union[Sequence[Callable], Callable] = None,
        add_eos: bool = True,
        add_bos: bool = True,
    ) -> None:
        super().__init__()

        validate(cuts)
        for cut in cuts:
            assert (
                len(cut.supervisions) == 1
            ), "Only the Cuts with single supervision are supported."

        self.cuts = cuts
        self.token_collater = TokenCollater(cuts, add_eos=add_eos, add_bos=add_bos)

        if feature_transforms is None:
            feature_transforms = []
        elif not isinstance(feature_transforms, Sequence):
            feature_transforms = [feature_transforms]

        assert all(isinstance(transform, Callable) for transform in feature_transforms), \
            "Feature transforms must be Callable"
        self.feature_transforms = feature_transforms

    def __getitem__(self, cut_ids: Iterable[str]) -> Dict[str, torch.Tensor]:
        cuts = self.cuts.subset(cut_ids=cut_ids)

        audio, audio_lens = collate_audio(cuts)
        features, features_lens = collate_features(cuts)

        for transform in self.feature_transforms:
            features = transform(features)

        tokens, tokens_lens = self.token_collater(cuts)

        return {
            "audio": audio,
            "features": features,
            "tokens": tokens,
            "audio_lens": audio_lens,
            "features_lens": features_lens,
            "tokens_lens": tokens_lens,
        }

    def __len__(self) -> int:
        return len(self.cuts)
