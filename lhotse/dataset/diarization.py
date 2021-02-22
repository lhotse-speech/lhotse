from typing import Dict, Iterable, Optional

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset

from lhotse import validate
from lhotse.cut import CutSet
from lhotse.dataset.collation import collate_features, collate_matrices


class DiarizationDataset(Dataset):
    """
    A PyTorch Dataset for the speaker diarization task.
    Our assumptions about speaker diarization are the following:

    * we assume a single channel input (for now), which could be either a true mono signal
        or a beamforming result from a microphone array.
    * we assume that the supervision used for model training is a speech activity matrix, with one
        row dedicated to each speaker (either in the current cut or the whole dataset,
        depending on the settings). The columns correspond to feature frames. Each row is effectively
        a Voice Activity Detection supervision for a single speaker. This setup is somewhat inspired by
        the TS-VAD paper: https://arxiv.org/abs/2005.07272

    Each item in this dataset is a dict of:

    .. code-block::

        {
            'features': (B x T x F) tensor
            'speaker_activity': (B x num_speaker x T) tensor
        }

    .. note: In cases when padding needs to be performed during collation,
        the cuts are silence-padded, and the speaker activity tensor is padded
        with CrossEntropyLoss().ignore_index.

    Constructor arguments:

    :param cuts: a ``CutSet`` used to create the dataset object.
    :param min_speaker_dim: optional int, when specified it will enforce that the matrix shape is at least
        that value (useful for datasets like CHiME 6 where the number of speakers is always 4, but some cuts
        might have less speakers than that).
    :param global_speaker_ids: a bool, indicates whether the same speaker should always retain the same row index
        in the speaker activity matrix (useful for speaker-dependent systems)
    :param root_dir: a prefix path to be attached to the feature files paths.
    """

    def __init__(
            self,
            cuts: CutSet,
            min_speaker_dim: Optional[int] = None,
            global_speaker_ids: bool = False,
    ) -> None:
        super().__init__()
        validate(cuts)
        self.cuts = cuts
        self.speakers = {spk: idx for idx, spk in enumerate(self.cuts.speakers)} if global_speaker_ids else None
        self.min_speaker_dim = min_speaker_dim

    def __getitem__(self, cut_ids: Iterable[str]) -> Dict[str, torch.Tensor]:
        cuts = self.cuts.subset(cut_ids=cut_ids)
        return {
            'features': collate_features(cuts),
            'speaker_activity': collate_matrices(
                (cut.speakers_feature_mask(
                    min_speaker_dim=self.min_speaker_dim,
                    speaker_to_idx_map=self.speakers,
                ) for cut in cuts),
                # In case padding is needed, we will add a special symbol
                # that tells the cross entropy loss to ignore the frame during scoring.
                padding_value=CrossEntropyLoss().ignore_index
            )
        }

    def __len__(self) -> int:
        return len(self.cuts)
