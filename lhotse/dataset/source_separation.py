from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import Dataset

from lhotse.cut import CutSet
from lhotse.utils import Pathlike

EPS = 1e-8


# TODO: add dynamic noise mixing to SourceSeparationDataset


class SourceSeparationDataset(Dataset):
    """
    A PyTorch Dataset for the source separation task. It's created from two CutSets - one provides the audio
    cuts for the sources, and the other one the audio cuts for the signal mix. When queried for data samples,
    it returns a dict of {'sources': tensor, 'mixture': tensor}.

    This Dataset performs on-the-fly feature-domain mixing of the sources. It expects the mixtures_set to contain
    MixedCuts, so that it knows which Cuts should be mixed together.
    """

    def __init__(
            self,
            sources_set: CutSet,
            mixtures_set: CutSet,
            root_dir: Optional[Pathlike] = None
    ):
        super().__init__()
        self.sources_set = sources_set
        self.mixtures_set = mixtures_set
        self.cut_ids = list(self.mixtures_set.cuts.keys())
        self.root_dir = Path(root_dir) if root_dir else None
        assert set(self.cut_ids) == set(self.mixtures_set.cuts.keys())

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        cut_id = self.cut_ids[idx]
        mixture_cut = self.mixtures_set.mixed_cuts[cut_id].with_cut_set(self.sources_set)
        source_cuts = [self.sources_set.cuts[track.cut_id] for track in mixture_cut.tracks]
        if len(source_cuts) != 2:
            raise NotImplementedError("Source separation for more than 2 sources is not yet supported.")
        mixture = torch.from_numpy(mixture_cut.load_features(root_dir=self.root_dir))
        sources = torch.stack(
            [torch.from_numpy(source_cut.load_features(root_dir=self.root_dir)) for source_cut in source_cuts],
            dim=0
        )

        # Compute the masks given the source features
        real_mask = sources / (sources.sum(1, keepdim=True) + EPS)
        # Get the src idx having the maximum energy
        binary_mask = real_mask.argmax(0)

        return {
            'sources': sources,
            'mixture': mixture,
            'real_mask': real_mask,
            'binary_mask': binary_mask
        }

    def __len__(self):
        return len(self.cut_ids)


class PreMixedSourceSeparationDataset(Dataset):
    """
    A PyTorch Dataset for the source separation task. It's created from two CutSets - one provides the audio
    cuts for the sources, and the other one the audio cuts for the signal mix. When queried for data samples,
    it returns a dict of {'sources': tensor, 'mixture': tensor}.

    It expects both CutSets to return regular Cuts, meaning that the signals were mixed in the time domain.
    In contrast to SourceSeparationDataset, no on-the-fly feature-domain-mixing is performed.
    """

    def __init__(
            self,
            sources_set: CutSet,
            mixtures_set: CutSet,
            root_dir: Optional[Pathlike] = None
    ):
        super().__init__()
        self.sources_set = sources_set
        self.mixtures_set = mixtures_set

        # TODO: The following code assumes that the speech separation dataset is created from
        #  cuts that span the whole recordings (i.e. one recording == one utterance).
        #  If we want to support datasets where cuts are parts of recordings (e.g. a single utterance in a
        #  15 minute conversation), we will need to provide an additional mapping here.
        self.mixture_to_source = {
            # We expect mixture and source cuts to share the same recording_ids
            cut.id: [c.id for c in self.sources_set if c.recording_id == cut.recording_id]
            for cut in self.mixtures_set
        }
        self.cut_ids = list(self.mixtures_set.cuts.keys())
        self.root_dir = Path(root_dir) if root_dir else None
        assert set(self.cut_ids) == set(self.mixtures_set.cuts.keys())

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        cut_id = self.cut_ids[idx]
        mixture_cut = self.mixtures_set.cuts[cut_id]
        source_cuts = [self.sources_set.cuts[id] for id in self.mixture_to_source[mixture_cut.id]]
        mixture = torch.from_numpy(mixture_cut.load_features(root_dir=self.root_dir))
        sources = torch.stack(
            [torch.from_numpy(source_cut.load_features(root_dir=self.root_dir)) for source_cut in source_cuts],
            dim=0
        )

        # Compute the masks given the source features
        real_mask = sources / (sources.sum(1, keepdim=True) + EPS)
        # Get the src idx having the maximum energy
        binary_mask = real_mask.argmax(0)

        return {
            'sources': sources,
            'mixture': mixture,
            'real_mask': real_mask,
            'binary_mask': binary_mask
        }

    def __len__(self):
        return len(self.cut_ids)
