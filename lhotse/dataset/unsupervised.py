from typing import Optional

import torch
from torch.utils.data import Dataset

from lhotse.cut import CutSet
from lhotse.features import FeatureExtractor
from lhotse.utils import Pathlike


class UnsupervisedDataset(Dataset):
    """
    Dataset that contains no supervision - it only provides the features extracted from recordings.
    The returned features are a :class:`torch.Tensor` of shape ``(T x F)``, where T is the number of frames,
    and F is the feature dimension.
    """

    def __init__(self, cuts: CutSet, root_dir: Optional[Pathlike] = None):
        super().__init__()
        self.cuts = cuts
        self.cut_ids = list(self.cuts.cuts.keys())
        self.root_dir = root_dir
        self._validate()

    def __getitem__(self, item: int) -> torch.Tensor:
        cut = self.cuts[self.cut_ids[item]]
        feats = cut.load_features(root_dir=self.root_dir)
        return torch.from_numpy(feats)

    def __len__(self):
        return len(self.cuts)

    def _validate(self):
        assert all(cut.has_features for cut in self.cuts)


class UnsupervisedWaveformDataset(UnsupervisedDataset):
    """
    A variant of UnsupervisedDataset that provides waveform samples instead of features.
    The output is a tensor of shape (C, T), with C being the number of channels and T the number of audio samples.
    In this implemenation, there will always be a single channel.
    """

    def __getitem__(self, item: int) -> torch.Tensor:
        cut = self.cuts[self.cut_ids[item]]
        audio = cut.load_audio(root_dir=self.root_dir)
        return torch.from_numpy(audio)

    def _validate(self):
        assert all(cut.has_recording for cut in self.cuts)


class DynamicUnsupervisedDataset(UnsupervisedDataset):
    """
    An example dataset that shows how to use on-the-fly feature extraction in Lhotse.
    It accepts an additional input - a FeatureExtractor.
    The output is approximately the same as that of the ``UnsupervisedDataset`` -
    there might be slight differences for ``MixedCut``s, because this dataset mixes them in the time domain,
    and ``UnsupervisedDataset`` does that in the feature domain.
    Cuts that are not mixed will yield identical results in both dataset classes.
    """

    def __init__(self, feature_extractor: FeatureExtractor, cuts: CutSet, root_dir: Optional[Pathlike] = None):
        super().__init__(cuts, root_dir)
        self.feature_extractor = feature_extractor

    def __getitem__(self, item: int) -> torch.Tensor:
        cut = self.cuts[self.cut_ids[item]]
        audio = cut.load_audio(root_dir=self.root_dir)
        features = self.feature_extractor.extract(audio, cut.sampling_rate)
        return torch.from_numpy(features)

    def _validate(self):
        assert all(cut.has_recording for cut in self.cuts)
