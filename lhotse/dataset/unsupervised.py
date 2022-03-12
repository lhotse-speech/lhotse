import warnings
from typing import Any, Dict, Optional

import torch

from lhotse import validate
from lhotse.audio import AudioLoadingError, DurationMismatchError
from lhotse.augmentation import AugmentFn
from lhotse.cut import CutSet
from lhotse.dataset.collation import collate_audio, collate_features, collate_matrices
from lhotse.features import FeatureExtractor
from lhotse.utils import NonPositiveEnergyError, suppress_and_warn


class UnsupervisedDataset(torch.utils.data.Dataset):
    """
    Dataset that contains no supervision - it only provides the features extracted from recordings.

    .. code-block::

        {
            'features': (B x T x F) tensor
            'features_lens': (B, ) tensor
        }
    """

    def __init__(self) -> None:
        super().__init__()

    def __getitem__(self, cuts: CutSet) -> Dict[str, Any]:
        self._validate(cuts)
        features, features_lens = collate_features(cuts)
        return {
            "cuts": cuts,
            "features": features,
            "features_lens": features_lens,
        }

    def _validate(self, cuts: CutSet) -> None:
        validate(cuts)
        assert all(cut.has_features for cut in cuts)


class UnsupervisedWaveformDataset(UnsupervisedDataset):
    """
    A variant of UnsupervisedDataset that provides waveform samples instead of features.
    The output is a tensor of shape (C, T), with C being the number of channels and T the number of audio samples.
    In this implementation, there will always be a single channel.

    Returns:

    .. code-block::

        {
            'audio': (B x NumSamples) float tensor
            'audio_lens': (B, ) int tensor
        }
    """

    def __init__(self, collate: bool = True) -> None:
        super().__init__()
        self.collate = collate

    def __getitem__(self, cuts: CutSet) -> Dict[str, Any]:
        if self.collate:
            audio, audio_lens = collate_audio(cuts)
            return {
                "cuts": cuts,
                "audio": audio,
                "audio_lens": audio_lens,
            }
        else:
            remain_cuts = []
            remain_audios = []
            for c in cuts:
                with suppress_and_warn(
                    AudioLoadingError, DurationMismatchError, NonPositiveEnergyError
                ):
                    remain_audios.append(c.load_audio())
                    remain_cuts.append(c)
            return {"cuts": CutSet.from_cuts(remain_cuts), "audio": remain_audios}

    def _validate(self, cuts: CutSet) -> None:
        validate(cuts)
        assert all(cut.has_recording for cut in cuts)


class DynamicUnsupervisedDataset(UnsupervisedDataset):
    """
    An example dataset that shows how to use on-the-fly feature extraction in Lhotse.
    It accepts two additional inputs - a FeatureExtractor and an optional WavAugmenter for time-domain data augmentation..
    The output is approximately the same as that of the ``UnsupervisedDataset`` -
    there might be slight differences for ``MixedCut``s, because this dataset mixes them in the time domain,
    and ``UnsupervisedDataset`` does that in the feature domain.
    Cuts that are not mixed will yield identical results in both dataset classes.
    """

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        augment_fn: Optional[AugmentFn] = None,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.augment_fn = augment_fn

    def __getitem__(self, cuts: CutSet) -> torch.Tensor:
        self._validate(cuts)

        def generate_cut(cuts: CutSet):
            for cut in cuts:
                with suppress_and_warn(
                    AudioLoadingError, DurationMismatchError, NonPositiveEnergyError
                ):
                    yield cut.compute_features(
                        extractor=self.feature_extractor,
                        augment_fn=self.augment_fn,
                    )

        features = collate_matrices(generate_cut(cuts))
        return features

    def _validate(self, cuts: CutSet) -> None:
        validate(cuts)
        assert all(cut.has_recording for cut in cuts)
