from typing import Callable, Dict, List, Sequence, Union

import numpy as np
import torch

from lhotse import validate
from lhotse.cut import CutSet
from lhotse.dataset.collation import TokenCollater, collate_audio
from lhotse.dataset.input_strategies import BatchIO, PrecomputedFeatures
from lhotse.utils import ifnone


class SpeechSynthesisDataset(torch.utils.data.Dataset):
    """
    The PyTorch Dataset for the speech synthesis task.
    Each item in this dataset is a dict of:

    .. code-block::

        {
            'audio': (B x NumSamples) float tensor
            'features': (B x NumFrames x NumFeatures) float tensor
            'tokens': (B x NumTokens) long tensor
            'audio_lens': (B, ) int tensor
            'features_lens': (B, ) int tensor
            'tokens_lens': (B, ) int tensor
        }
    """

    def __init__(
        self,
        cuts: CutSet,
        cut_transforms: List[Callable[[CutSet], CutSet]] = None,
        feature_input_strategy: BatchIO = PrecomputedFeatures(),
        feature_transforms: Union[Sequence[Callable], Callable] = None,
        speaker_ids: List[str] = None,
        add_eos: bool = True,
        add_bos: bool = True,
    ) -> None:
        super().__init__()

        self.cuts = cuts
        self.token_collater = TokenCollater(cuts, add_eos=add_eos, add_bos=add_bos)
        self.cut_transforms = ifnone(cut_transforms, [])
        self.feature_input_strategy = feature_input_strategy
        self.speaker_ids = speaker_ids
        if speaker_ids is not None:
            self.sid_to_onehot_map = get_sid_to_onehot_map(speaker_ids)

        if feature_transforms is None:
            feature_transforms = []
        elif not isinstance(feature_transforms, Sequence):
            feature_transforms = [feature_transforms]

        assert all(
            isinstance(transform, Callable) for transform in feature_transforms
        ), "Feature transforms must be Callable"
        self.feature_transforms = feature_transforms

    def __getitem__(self, cuts: CutSet) -> Dict[str, torch.Tensor]:
        validate_for_tts(cuts)

        for transform in self.cut_transforms:
            cuts = transform(cuts)

        audio, audio_lens = collate_audio(cuts)
        features, features_lens = self.feature_input_strategy(cuts)

        for transform in self.feature_transforms:
            features = transform(features)

        tokens, tokens_lens = self.token_collater(cuts)
        if self.speaker_ids is not None:
            speakers = torch.tensor(
                [self.sid_to_onehot_map[cut.supervisions[0].speaker] for cut in cuts]
            )

        return {
            "audio": audio,
            "features": features,
            "tokens": tokens,
            "audio_lens": audio_lens,
            "features_lens": features_lens,
            "tokens_lens": tokens_lens,
            "speakers": speakers,
        }


def validate_for_tts(cuts: CutSet) -> None:
    validate(cuts)
    for cut in cuts:
        assert (
            len(cut.supervisions) == 1
        ), "Only the Cuts with single supervision are supported."


def get_sid_to_onehot_map(sid_list) -> Dict[str, np.ndarray]:
    sid_to_onehot_map = {}
    for index, sid in enumerate(sid_list):
        sid_onehot_vector = np.zeros(len(sid_list))
        sid_onehot_vector[index] = 1
        sid_to_onehot_map[sid] = sid_onehot_vector
    return sid_to_onehot_map
