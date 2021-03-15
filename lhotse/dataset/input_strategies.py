import logging
from typing import Dict, Tuple

import torch

from lhotse import CutSet, FeatureExtractor
from lhotse.cut import compute_supervisions_frame_mask
from lhotse.dataset.collation import collate_audio, collate_features, collate_vectors
from lhotse.utils import compute_num_frames, supervision_to_frames, supervision_to_samples


class InputStrategy:
    """
    Converts a :class:`CutSet` into a collated batch of audio representations.
    These representations can be e.g. audio samples or features.
    They might also be single or multi channel.

    This is a base class that only defines the interface.
    """

    def __call__(self, cuts: CutSet) -> Tuple[torch.Tensor, torch.IntTensor]:
        raise NotImplementedError()

    def supervision_intervals(self, cuts: CutSet) -> Dict[str, torch.Tensor]:
        """
        Returns a dict that specifies the start and end bounds for each supervision,
        as a 1-D int tensor.

        Depending on the strategy, the dict should look like:

        .. code-block:

            {
                "start_frame": tensor(shape=(S,)),
                "num_frames": tensor(shape=(S,))
            }

        or

        .. code-block:

            {
                "start_sample": tensor(shape=(S,)),
                "num_samples": tensor(shape=(S,))
            }

        Where ``S`` is the total number of supervisions encountered in the :class:`CutSet`.
        Note that ``S`` might be different than the number of cuts (``B``).
        """
        raise NotImplementedError()

    def supervision_masks(self, cuts: CutSet) -> torch.Tensor:
        """
        Returns a collated batch of masks, marking the supervised regions in cuts.
        They are zero-padded to the longest cut.

        Depending on the strategy implementation, it is expected to be a
        tensor of shape ``(B, NF)`` or ``(B, NS)``, where ``B`` denotes the number of cuts,
        ``NF`` the number of frames and ``NS`` the total number of samples.
        ``NF`` and ``NS`` are determined by the longest cut in a batch.
        """
        raise NotImplementedError()


class PrecomputedFeatures(InputStrategy):
    """
    :class:`InputStrategy` that reads pre-computed features, whose manifests
    are attached to cuts, from disk.

    It pads the feature matrices, if needed.
    """

    def __call__(self, cuts: CutSet) -> Tuple[torch.Tensor, torch.IntTensor]:
        return collate_features(cuts)

    def supervision_intervals(self, cuts: CutSet) -> Dict[str, torch.Tensor]:
        """
        Returns a dict that specifies the start and end bounds for each supervision,
        as a 1-D int tensor, in terms of frames.
        """
        start_frames, nums_frames = zip(*(
            supervision_to_frames(sup, cut.frame_shift, cut.sampling_rate, max_frames=cut.num_frames)
            for cut in cuts
            for sup in cut.supervisions
        ))
        return {
            'start_frame': torch.tensor(start_frames, dtype=torch.int32),
            'num_frames': torch.tensor(nums_frames, dtype=torch.int32)
        }

    def supervision_masks(self, cuts: CutSet) -> torch.Tensor:
        """Returns the mask for supervised frames."""
        return collate_vectors([cut.supervisions_feature_mask() for cut in cuts])


class AudioSamples(InputStrategy):
    """
    :class:`InputStrategy` that reads single-channel recordings, whose manifests
    are attached to cuts, from disk (or other audio source).

    It pads the recordings, if needed.
    """

    def __call__(self, cuts: CutSet) -> Tuple[torch.Tensor, torch.IntTensor]:
        return collate_audio(cuts)

    def supervision_intervals(self, cuts: CutSet) -> Dict[str, torch.Tensor]:
        """
        Returns a dict that specifies the start and end bounds for each supervision,
        as a 1-D int tensor, in terms of samples.
        """
        start_samples, nums_samples = zip(*(
            supervision_to_samples(sup, cut.sampling_rate)
            for cut in cuts
            for sup in cut.supervisions
        ))
        return {
            'start_sample': torch.tensor(start_samples, dtype=torch.int32),
            'num_samples': torch.tensor(nums_samples, dtype=torch.int32)
        }

    def supervision_masks(self, cuts: CutSet) -> torch.Tensor:
        """Returns the mask for supervised samples."""
        return collate_vectors([cut.supervisions_audio_mask() for cut in cuts])


class OnTheFlyFeatures(InputStrategy):
    """
    :class:`InputStrategy` that reads single-channel recordings, whose manifests
    are attached to cuts, from disk (or other audio source).
    Then, it uses a :class:`FeatureExtractor` to compute their features on-the-fly.

    It pads the recordings, if needed.

    .. note:
        The batch feature extraction performed here is not as efficient as it could be,
        but it allows to use arbitrary feature extraction method that may work on
        a single recording at a time.
    """

    def __init__(self, extractor: FeatureExtractor):
        self.extractor = extractor

    def __call__(self, cuts: CutSet) -> Tuple[torch.Tensor, torch.IntTensor]:
        audio, _ = collate_audio(cuts)

        features_single = []
        for idx, cut in enumerate(cuts):
            samples = audio[idx].numpy()
            try:
                features = self.extractor.extract(samples, cuts[idx].sampling_rate)
            except:
                logging.error(f"Error while extracting the features for cut with ID {cut.id} -- details:\n{cut}")
                raise
            features_single.append(torch.from_numpy(features))
        features_batch = torch.stack(features_single)

        feature_lens = torch.tensor([
            compute_num_frames(
                cut.duration,
                self.extractor.frame_shift,
                cut.sampling_rate
            ) for cut in cuts
        ], dtype=torch.int32)

        return features_batch, feature_lens

    def supervision_intervals(self, cuts: CutSet) -> Dict[str, torch.Tensor]:
        """
        Returns a dict that specifies the start and end bounds for each supervision,
        as a 1-D int tensor, in terms of frames.
        """
        start_frames, nums_frames = zip(*(
            supervision_to_frames(sup, self.extractor.frame_shift, cut.sampling_rate)
            for cut in cuts
            for sup in cut.supervisions
        ))
        return {
            'start_frame': torch.tensor(start_frames, dtype=torch.int32),
            'num_frames': torch.tensor(nums_frames, dtype=torch.int32)
        }

    def supervision_masks(self, cuts: CutSet) -> torch.Tensor:
        """Returns the mask for supervised samples."""
        return collate_vectors([compute_supervisions_frame_mask(cut, self.extractor.frame_shift) for cut in cuts])
