import logging
from typing import Callable, Dict, List, Tuple, Optional

import torch

from lhotse import CutSet, FeatureExtractor
from lhotse.cut import compute_supervisions_frame_mask
from lhotse.dataset.collation import collate_audio, collate_features, collate_posts, collate_vectors
from lhotse.utils import compute_num_frames, ifnone, supervision_to_frames, supervision_to_samples


class InputStrategy:
    """
    Converts a :class:`CutSet` into a collated batch of audio representations.
    These representations can be e.g. audio samples or features.
    They might also be single or multi channel.

    This is a base class that only defines the interface.

    .. automethod:: __call__
    """

    def __call__(self, cuts: CutSet) -> Tuple[torch.Tensor, torch.IntTensor]:
        """Returns a tensor with collated input signals, and a tensor of length of each signal before padding."""
        raise NotImplementedError()

    def supervision_intervals(self, cuts: CutSet) -> Dict[str, torch.Tensor]:
        """
        Returns a dict that specifies the start and end bounds for each supervision,
        as a 1-D int tensor.

        Depending on the strategy, the dict should look like:

        .. code-block:

            {
                "sequence_idx": tensor(shape=(S,)),
                "start_frame": tensor(shape=(S,)),
                "num_frames": tensor(shape=(S,)),
            }

        or

        .. code-block:

            {
                "sequence_idx": tensor(shape=(S,)),
                "start_sample": tensor(shape=(S,)),
                "num_samples": tensor(shape=(S,))
            }

        Where ``S`` is the total number of supervisions encountered in the :class:`CutSet`.
        Note that ``S`` might be different than the number of cuts (``B``).
        ``sequence_idx`` means the index of the corresponding feature matrix (or cut) in a batch.
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
    :class:`PrecomputedFeatures` that reads pre-computed features, whose manifests
    are attached to cuts, from disk.

    It pads the feature matrices, if needed.

    .. automethod:: __call__
    """

    def __call__(self, cuts: CutSet) -> Tuple[torch.Tensor, torch.IntTensor]:
        """
        Reads the pre-computed features from disk/other storage.
        The returned shape is ``(B, T, F) => (batch_size, num_frames, num_features)``.

        :return: a tensor with collated features, and a tensor of ``num_frames`` of each cut before padding."""
        return collate_features(cuts)

    def supervision_intervals(self, cuts: CutSet) -> Dict[str, torch.Tensor]:
        """
        Returns a dict that specifies the start and end bounds for each supervision,
        as a 1-D int tensor, in terms of frames:

        .. code-block:

            {
                "sequence_idx": tensor(shape=(S,)),
                "start_frame": tensor(shape=(S,)),
                "num_frames": tensor(shape=(S,))
            }

        Where ``S`` is the total number of supervisions encountered in the :class:`CutSet`.
        Note that ``S`` might be different than the number of cuts (``B``).
        ``sequence_idx`` means the index of the corresponding feature matrix (or cut) in a batch.
        """
        start_frames, nums_frames = zip(*(
            supervision_to_frames(sup, cut.frame_shift, cut.sampling_rate, max_frames=cut.num_frames)
            for cut in cuts
            for sup in cut.supervisions
        ))
        sequence_idx = [i for i, c in enumerate(cuts) for s in c.supervisions]
        return {
            'sequence_idx': torch.tensor(sequence_idx, dtype=torch.int32),
            'start_frame': torch.tensor(start_frames, dtype=torch.int32),
            'num_frames': torch.tensor(nums_frames, dtype=torch.int32)
        }

    def supervision_masks(self, cuts: CutSet, use_alignment_if_exists: Optional[str] = None) -> torch.Tensor:
        """Returns the mask for supervised frames.
        :param use_alignment_if_exists: optional str, key for alignment type to use for generating the mask. If not
            exists, fall back on supervision time spans.
        """
        return collate_vectors([cut.supervisions_feature_mask(use_alignment_if_exists=use_alignment_if_exists) for cut in cuts])


class PrecomputedPosteriors(InputStrategy):
    """
    :class:`PrecomputedPosteriors` that reads pre-computed posteriors, whose manifests
    are attached to cuts, from disk.

    It pads the posterior matrices, if needed.

    .. automethod:: __call__
    """

    def __call__(self, cuts: CutSet) -> Tuple[torch.Tensor, torch.IntTensor]:
        """
        Reads the pre-computed posteriors from disk/other storage.
        The returned shape is ``(B, T, F) => (batch_size, num_frames, posts_dim)``.

        :return: a tensor with collated posteriors, and a tensor of ``num_frames`` of each cut before padding."""
        return collate_posts(cuts)

    def supervision_intervals(self, cuts: CutSet) -> Dict[str, torch.Tensor]:
        """
        Returns a dict that specifies the start and end bounds for each supervision,
        as a 1-D int tensor, in terms of frames:

        .. code-block:

            {
                "sequence_idx": tensor(shape=(S,)),
                "start_frame": tensor(shape=(S,)),
                "num_frames": tensor(shape=(S,))
            }

        Where ``S`` is the total number of supervisions encountered in the :class:`CutSet`.
        Note that ``S`` might be different than the number of cuts (``B``).
        ``sequence_idx`` means the index of the corresponding feature matrix (or cut) in a batch.
        """
        start_frames, nums_frames = zip(*(
            supervision_to_frames(sup, cut.frame_shift, cut.sampling_rate, max_frames=cut.num_frames)
            for cut in cuts
            for sup in cut.supervisions
        ))
        sequence_idx = [i for i, c in enumerate(cuts) for s in c.supervisions]

        first_cut = next(iter(cuts))
        # we assume that all posteriors have the same subsampling factor
        sf = first_cut.posts.subsampling_factor
        return {
            'sequence_idx': torch.tensor(sequence_idx, dtype=torch.int32),
            'start_frame': torch.tensor(start_frames, dtype=torch.int32) // sf,
            'num_frames': torch.tensor(nums_frames, dtype=torch.int32)  // sf
        }


class AudioSamples(InputStrategy):
    """
    :class:`InputStrategy` that reads single-channel recordings, whose manifests
    are attached to cuts, from disk (or other audio source).

    It pads the recordings, if needed.

    .. automethod:: __call__
    """

    def __call__(self, cuts: CutSet) -> Tuple[torch.Tensor, torch.IntTensor]:
        """
        Reads the audio samples from recordings on disk/other storage.
        The returned shape is ``(B, T) => (batch_size, num_samples)``.

        :return: a tensor with collated audio samples, and a tensor of ``num_samples`` of each cut before padding.
        """
        return collate_audio(cuts)

    def supervision_intervals(self, cuts: CutSet) -> Dict[str, torch.Tensor]:
        """
        Returns a dict that specifies the start and end bounds for each supervision,
        as a 1-D int tensor, in terms of samples:

        .. code-block:

            {
                "sequence_idx": tensor(shape=(S,)),
                "start_sample": tensor(shape=(S,)),
                "num_samples": tensor(shape=(S,))
            }

        Where ``S`` is the total number of supervisions encountered in the :class:`CutSet`.
        Note that ``S`` might be different than the number of cuts (``B``).
        ``sequence_idx`` means the index of the corresponding feature matrix (or cut) in a batch.

        """
        start_samples, nums_samples = zip(*(
            supervision_to_samples(sup, cut.sampling_rate)
            for cut in cuts
            for sup in cut.supervisions
        ))
        sequence_idx = [i for i, c in enumerate(cuts) for s in c.supervisions]
        return {
            'sequence_idx': torch.tensor(sequence_idx, dtype=torch.int32),
            'start_sample': torch.tensor(start_samples, dtype=torch.int32),
            'num_samples': torch.tensor(nums_samples, dtype=torch.int32)
        }

    def supervision_masks(self, cuts: CutSet, use_alignment_if_exists: Optional[str] = None) -> torch.Tensor:
        """Returns the mask for supervised samples.
        :param use_alignment_if_exists: optional str, key for alignment type to use for generating the mask. If not
            exists, fall back on supervision time spans.
        """
        return collate_vectors([cut.supervisions_audio_mask(use_alignment_if_exists=use_alignment_if_exists) for cut in cuts])


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

    .. automethod:: __call__
    """

    def __init__(
            self,
            extractor: FeatureExtractor,
            wave_transforms: List[Callable[[torch.Tensor], torch.Tensor]] = None
    ) -> None:
        """
        OnTheFlyFeatures' constructor.

        :param extractor: the feature extractor used on-the-fly (individually on each waveform).
        :param wave_transforms: an optional list of transforms applied on the batch of audio
            waveforms collated into a single tensor, right before the feature extraction.
        """
        self.extractor = extractor
        self.wave_transforms = ifnone(wave_transforms, [])

    def __call__(self, cuts: CutSet) -> Tuple[torch.Tensor, torch.IntTensor]:
        """
        Reads the audio samples from recordings on disk/other storage
        and computes their features.
        The returned shape is ``(B, T, F) => (batch_size, num_frames, num_features)``.

        :return: a tensor with collated features, and a tensor of ``num_frames`` of each cut before padding.
        """
        audio, _ = collate_audio(cuts)

        for tfnm in self.wave_transforms:
            audio = tfnm(audio)

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
        as a 1-D int tensor, in terms of frames:

        .. code-block:

            {
                "sequence_idx": tensor(shape=(S,)),
                "start_frame": tensor(shape=(S,)),
                "num_frames": tensor(shape=(S,))
            }

        Where ``S`` is the total number of supervisions encountered in the :class:`CutSet`.
        Note that ``S`` might be different than the number of cuts (``B``).
        ``sequence_idx`` means the index of the corresponding feature matrix (or cut) in a batch.
        """
        start_frames, nums_frames = zip(*(
            supervision_to_frames(sup, self.extractor.frame_shift, cut.sampling_rate)
            for cut in cuts
            for sup in cut.supervisions
        ))
        sequence_idx = [i for i, c in enumerate(cuts) for s in c.supervisions]
        return {
            'sequence_idx': torch.tensor(sequence_idx, dtype=torch.int32),
            'start_frame': torch.tensor(start_frames, dtype=torch.int32),
            'num_frames': torch.tensor(nums_frames, dtype=torch.int32)
        }

    def supervision_masks(self, cuts: CutSet, use_alignment_if_exists: Optional[str] = None) -> torch.Tensor:
        """Returns the mask for supervised samples.
        :param use_alignment_if_exists: optional str, key for alignment type to use for generating the mask. If not
            exists, fall back on supervision time spans.
        """
        return collate_vectors(
            [
                compute_supervisions_frame_mask(
                    cut,
                    frame_shift=self.extractor.frame_shift,
                    use_alignment_if_exists=use_alignment_if_exists
                ) for cut in cuts
            ]
        )
