from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from lhotse.cut import AnyCut, Cut, CutSet
from lhotse.utils import EPSILON


class SourceSeparationDataset(Dataset):
    """
    An abstract base class, implementing PyTorch Dataset for the source separation task.
    It's created from two CutSets - one provides the audio cuts for the sources, and the other one the audio cuts for
    the signal mix. When queried for data samples, it returns a dict of:

    .. code-block::

        {
            'sources': (N x T x F) tensor,
            'mixture': (T x F) tensor,
            'real_mask': (N x T x F) tensor,
            'binary_mask': (T x F) tensor
        }
    """

    def __init__(
            self,
            sources_set: CutSet,
            mixtures_set: CutSet,
    ):
        super().__init__()
        self.sources_set = sources_set
        self.mixtures_set = mixtures_set
        self.cut_ids = list(self.mixtures_set.ids)

    def _obtain_mixture(self, cut_id: str) -> Tuple[AnyCut, List[Cut]]:
        raise NotImplementedError("You are using SpeechSeparationDataset, which is an abstract base class; instead, "
                                  "use one of its derived classes that specify whether the mix is pre-computed or "
                                  "done dynamically (on-the-fly).")

    def validate(self):
        # Make sure it's possible to iterate through the whole dataset and resolve the sources for each mixture
        for cut in self.mixtures_set.mixed_cuts.values():
            _, source_cuts = self._obtain_mixture(cut.id)
            assert len(source_cuts) > 1

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        cut_id = self.cut_ids[idx]
        mixture_cut, source_cuts = self._obtain_mixture(cut_id=cut_id)

        mixture = torch.from_numpy(mixture_cut.load_features())
        sources = torch.stack(
            [torch.from_numpy(source_cut.load_features()) for source_cut in source_cuts],
            dim=0
        )

        # Compute the masks given the source features
        sources_exp = sources.exp()
        real_mask = sources_exp / (sources_exp.sum(0, keepdim=True) + EPSILON)
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


class DynamicallyMixedSourceSeparationDataset(SourceSeparationDataset):
    """
    A PyTorch Dataset for the source separation task.
    It's created from a number of CutSets:

    - ``sources_set``: provides the audio cuts for the sources that (the targets of source separation),
    - ``mixtures_set``: provides the audio cuts for the signal mix (the input of source separation),
    - ``nonsources_set``: *(optional)* provides the audio cuts for other signals that are in the mix,
      but are not the targets of source separation. Useful for adding noise.

    When queried for data samples, it returns a dict of:

    .. code-block::

        {
            'sources': (N x T x F) tensor,
            'mixture': (T x F) tensor,
            'real_mask': (N x T x F) tensor,
            'binary_mask': (T x F) tensor
        }

    This Dataset performs on-the-fly feature-domain mixing of the sources. It expects the mixtures_set to contain
    MixedCuts, so that it knows which Cuts should be mixed together.
    """

    def __init__(
            self,
            sources_set: CutSet,
            mixtures_set: CutSet,
            nonsources_set: Optional[CutSet] = None,
    ):
        super().__init__(sources_set=sources_set, mixtures_set=mixtures_set)
        self.nonsources_set = nonsources_set

    def _obtain_mixture(self, cut_id: str) -> Tuple[AnyCut, List[Cut]]:
        mixture_cut = self.mixtures_set.mixed_cuts[cut_id]
        source_cuts = [
            track.cut
            for track in mixture_cut.tracks
            if track.cut.id in self.sources_set  # tracks will be missing in the sources set when they are noise
        ]
        return mixture_cut, source_cuts


class PreMixedSourceSeparationDataset(SourceSeparationDataset):
    """
    A PyTorch Dataset for the source separation task.
    It's created from two CutSets - one provides the audio cuts for the sources, and the other one the audio cuts for
    the signal mix. When queried for data samples, it returns a dict of:

    .. code-block::

        {
            'sources': (N x T x F) tensor,
            'mixture': (T x F) tensor,
            'real_mask': (N x T x F) tensor,
            'binary_mask': (T x F) tensor
        }

    It expects both CutSets to return regular Cuts, meaning that the signals were mixed in the time domain.
    In contrast to DynamicallyMixedSourceSeparationDataset, no on-the-fly feature-domain-mixing is performed.
    """

    def __init__(
            self,
            sources_set: CutSet,
            mixtures_set: CutSet,
    ):
        # The following code assumes that the speech separation dataset is created from
        # cuts that span the whole recordings (i.e. one recording == one utterance), so it is safe to assume that
        # matching them by recording_id will yield correct mixture <=> sources mapping.
        # If we want to support datasets where cuts are parts of recordings (e.g. a single utterance in a
        # 15 minute conversation), we will need to provide an external mapping here.
        self.mixture_to_source = {
            # We expect mixture and source cuts to share the same recording_ids
            cut.id: [c.id for c in sources_set if c.recording_id == cut.recording_id]
            for cut in mixtures_set
        }
        super().__init__(sources_set=sources_set, mixtures_set=mixtures_set)

    def _obtain_mixture(self, cut_id: str) -> Tuple[AnyCut, List[Cut]]:
        mixture_cut = self.mixtures_set.cuts[cut_id]
        source_cuts = [self.sources_set.cuts[id] for id in self.mixture_to_source[mixture_cut.id]]
        return mixture_cut, source_cuts
