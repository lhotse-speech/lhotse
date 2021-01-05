from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import Dataset

from lhotse import validate
from lhotse.cut import CutSet
from lhotse.utils import Pathlike


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
            'features': (T x F) tensor
            'speaker_activity': (num_speaker x T) tensor
        }

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
    ):
        super().__init__()
        validate(cuts)
        self.cuts = cuts
        self.cut_ids = list(cuts.ids)
        self.speakers = {spk: idx for idx, spk in enumerate(self.cuts.speakers)} if global_speaker_ids else None
        self.min_speaker_dim = min_speaker_dim

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        cut_id = self.cut_ids[idx]
        cut = self.cuts[cut_id]

        features = torch.from_numpy(cut.load_features())
        speaker_activity_matrix = torch.from_numpy(cut.speakers_feature_mask(
            min_speaker_dim=self.min_speaker_dim,
            speaker_to_idx_map=self.speakers
        ))

        return {
            'features': features,
            'speaker_activity': speaker_activity_matrix
        }

    def __len__(self) -> int:
        return len(self.cut_ids)
