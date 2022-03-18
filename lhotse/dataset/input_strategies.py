import logging
from concurrent.futures import Executor, ThreadPoolExecutor
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

import torch

from lhotse import CutSet, FeatureExtractor
from lhotse.cut import compute_supervisions_frame_mask
from lhotse.dataset.collation import (
    collate_audio,
    collate_features,
    collate_matrices,
    collate_vectors,
    read_audio_from_cuts,
)
from lhotse.utils import (
    LOG_EPSILON,
    compute_num_frames,
    ifnone,
    supervision_to_frames,
    supervision_to_samples,
)

ExecutorType = TypeVar("ExecutorType", bound=Executor)


class BatchIO:
    """
    Converts a :class:`CutSet` into a collated batch of audio representations.
    These representations can be e.g. audio samples or features.
    They might also be single or multi channel.

    All InputStrategies support the ``executor`` parameter in the constructor.
    It allows to pass a ``ThreadPoolExecutor`` or a ``ProcessPoolExecutor``
    to parallelize reading audio/features from wherever they are stored.
    Note that this approach is incompatible with specifying the ``num_workers``
    to ``torch.utils.data.DataLoader``, but in some instances may be faster.

    .. note:: This is a base class that only defines the interface.

    .. automethod:: __call__
    """

    def __init__(
        self,
        num_workers: int = 0,
        executor_type: Type[ExecutorType] = ThreadPoolExecutor,
    ) -> None:
        self.num_workers = num_workers
        self._executor_type = executor_type

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


class PrecomputedFeatures(BatchIO):
    """
    :class:`InputStrategy` that reads pre-computed features, whose manifests
    are attached to cuts, from disk.

    It automatically pads the feature matrices so that every example has the same number
    of frames as the longest cut in a mini-batch.
    This is needed to put all examples into a single tensor.
    The padding value is a low log-energy, around log(1e-10).

    .. automethod:: __call__
    """

    def __call__(self, cuts: CutSet) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reads the pre-computed features from disk/other storage.
        The returned shape is ``(B, T, F) => (batch_size, num_frames, num_features)``.

        :return: a tensor with collated features, and a tensor of ``num_frames`` of each cut before padding."""
        return collate_features(
            cuts,
            executor=_get_executor(self.num_workers, executor_type=self._executor_type),
        )

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
        start_frames, nums_frames = zip(
            *(
                supervision_to_frames(
                    sup, cut.frame_shift, cut.sampling_rate, max_frames=cut.num_frames
                )
                for cut in cuts
                for sup in cut.supervisions
            )
        )
        sequence_idx = [i for i, c in enumerate(cuts) for s in c.supervisions]
        return {
            "sequence_idx": torch.tensor(sequence_idx, dtype=torch.int32),
            "start_frame": torch.tensor(start_frames, dtype=torch.int32),
            "num_frames": torch.tensor(nums_frames, dtype=torch.int32),
        }

    def supervision_masks(
        self, cuts: CutSet, use_alignment_if_exists: Optional[str] = None
    ) -> torch.Tensor:
        """Returns the mask for supervised frames.

        :param use_alignment_if_exists: optional str, key for alignment type to use for generating the mask. If not
            exists, fall back on supervision time spans.
        """
        return collate_vectors(
            [
                cut.supervisions_feature_mask(
                    use_alignment_if_exists=use_alignment_if_exists
                )
                for cut in cuts
            ]
        )


class AudioSamples(BatchIO):
    """
    :class:`InputStrategy` that reads single-channel recordings, whose manifests
    are attached to cuts, from disk (or other audio source).

    It automatically zero-pads the recordings so that every example has the same number
    of audio samples as the longest cut in a mini-batch.
    This is needed to put all examples into a single tensor.

    .. automethod:: __call__
    """

    def __init__(
        self,
        num_workers: int = 0,
        fault_tolerant: bool = False,
        executor_type: Type[ExecutorType] = ThreadPoolExecutor,
    ) -> None:
        """
        AudioSamples constructor.

        :param num_workers: when larger than 0, we will spawn an executor (of type specified
            by ``executor_type``) to read the audio data in parallel.
            Thread executor can be used with PyTorch's DataLoader, whereas Process executor
            would fail (but could be faster for other applications).
        :param fault_tolerant: when ``True``, the cuts for which audio loading failed
            will be skipped. It will make ``__call__`` return an additional item,
            which is the CutSet for which we successfully read the audio.
            It may be a subset of the input CutSet.
        :param executor_type: the type of executor used for parallel audio reads
            (only relevant when ``num_workers>0``).
        """
        super().__init__(num_workers=num_workers, executor_type=executor_type)
        self.fault_tolerant = fault_tolerant

    def __call__(
        self, cuts: CutSet
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, CutSet]
    ]:
        """
        Reads the audio samples from recordings on disk/other storage.
        The returned shape is ``(B, T) => (batch_size, num_samples)``.

        :return: a tensor with collated audio samples, and a tensor of ``num_samples`` of each cut before padding.
        """
        return collate_audio(
            cuts,
            executor=_get_executor(self.num_workers, executor_type=self._executor_type),
            fault_tolerant=self.fault_tolerant,
        )

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
        start_samples, nums_samples = zip(
            *(
                supervision_to_samples(sup, cut.sampling_rate)
                for cut in cuts
                for sup in cut.supervisions
            )
        )
        sequence_idx = [i for i, c in enumerate(cuts) for s in c.supervisions]
        return {
            "sequence_idx": torch.tensor(sequence_idx, dtype=torch.int32),
            "start_sample": torch.tensor(start_samples, dtype=torch.int32),
            "num_samples": torch.tensor(nums_samples, dtype=torch.int32),
        }

    def supervision_masks(
        self, cuts: CutSet, use_alignment_if_exists: Optional[str] = None
    ) -> torch.Tensor:
        """Returns the mask for supervised samples.

        :param use_alignment_if_exists: optional str, key for alignment type to use for generating the mask. If not
            exists, fall back on supervision time spans.
        """
        return collate_vectors(
            [
                cut.supervisions_audio_mask(
                    use_alignment_if_exists=use_alignment_if_exists
                )
                for cut in cuts
            ]
        )


class OnTheFlyFeatures(BatchIO):
    """
    :class:`InputStrategy` that reads single-channel recordings, whose manifests
    are attached to cuts, from disk (or other audio source).
    Then, it uses a :class:`FeatureExtractor` to compute their features on-the-fly.

    It automatically pads the feature matrices so that every example has the same number
    of frames as the longest cut in a mini-batch.
    This is needed to put all examples into a single tensor.
    The padding value is a low log-energy, around log(1e-10).

    .. note:
        The batch feature extraction performed here is not as efficient as it could be,
        but it allows to use arbitrary feature extraction method that may work on
        a single recording at a time.

    .. automethod:: __call__
    """

    def __init__(
        self,
        extractor: FeatureExtractor,
        wave_transforms: List[Callable[[torch.Tensor], torch.Tensor]] = None,
        num_workers: int = 0,
        use_batch_extract: bool = True,
        fault_tolerant: bool = False,
        return_audio: bool = False,
        executor_type: Type[ExecutorType] = ThreadPoolExecutor,
    ) -> None:
        """
        OnTheFlyFeatures' constructor.

        :param extractor: the feature extractor used on-the-fly (individually on each waveform).
        :param wave_transforms: an optional list of transforms applied on the batch of audio
            waveforms collated into a single tensor, right before the feature extraction.
        :param num_workers: when larger than 0, we will spawn an executor (of type specified
            by ``executor_type``) to read the audio data in parallel.
            Thread executor can be used with PyTorch's DataLoader, whereas Process executor
            would fail (but could be faster for other applications).
        :param use_batch_extract: when ``True``, we will call
            :meth:`~lhotse.features.base.FeatureExtractor.extract_batch` to compute the features
            as it is possibly faster. It has a restriction that all cuts must have the same
            sampling rate. If that is not the case, set this to ``False``.
        :param fault_tolerant: when ``True``, the cuts for which audio loading failed
            will be skipped. It will make ``__call__`` return an additional item,
            which is the CutSet for which we successfully read the audio.
            It may be a subset of the input CutSet.
        :param return_audio: When ``True``, calling this object will additionally return collated
            audio tensor and audio lengths tensor.
        :param executor_type: the type of executor used for parallel audio reads
            (only relevant when ``num_workers>0``).
        """
        super().__init__(num_workers=num_workers, executor_type=executor_type)
        self.extractor = extractor
        self.wave_transforms = ifnone(wave_transforms, [])
        self.use_batch_extract = use_batch_extract
        self.fault_tolerant = fault_tolerant
        self.return_audio = return_audio

    def __call__(
        self, cuts: CutSet
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, CutSet]
    ]:
        """
        Reads the audio samples from recordings on disk/other storage
        and computes their features.
        The returned shape is ``(B, T, F) => (batch_size, num_frames, num_features)``.

        :return: a tuple of objcets: ``(feats, feat_lens, [audios, audio_lens], [cuts])``.
            Tensors ``audios`` and ``audio_lens`` are returned when ``return_audio=True``.
            CutSet ``cuts`` is returned when ``fault_tolerant=True``.
        """
        audios, cuts = read_audio_from_cuts(
            cuts,
            executor=_get_executor(self.num_workers, executor_type=self._executor_type),
            suppress_errors=self.fault_tolerant,
        )

        for tfnm in self.wave_transforms:
            for idx in range(len(audios)):
                audios[idx] = tfnm(audios[idx])

        if self.use_batch_extract:
            # Batch extraction is possibly faster depending on the implementation
            # of the feature extractor.
            assert all(c.sampling_rate == cuts[0].sampling_rate for c in cuts)
            features_single = self.extractor.extract_batch(
                audios, sampling_rate=cuts[0].sampling_rate
            )
        else:
            # Sequential extraction allows the sampling rates to be different.
            features_single = []
            for idx, cut in enumerate(cuts):
                samples = audios[idx].numpy()
                try:
                    features = self.extractor.extract(samples, cuts[idx].sampling_rate)
                except:
                    logging.error(
                        f"Error while extracting the features for cut with ID {cut.id} -- details:\n{cut}"
                    )
                    raise
                features_single.append(torch.from_numpy(features))

        features_batch = collate_matrices(features_single, padding_value=LOG_EPSILON)

        feature_lens = torch.tensor(
            [
                compute_num_frames(
                    cut.duration, self.extractor.frame_shift, cut.sampling_rate
                )
                for cut in cuts
            ],
            dtype=torch.int64,
        )

        out = (features_batch, feature_lens)

        if self.return_audio:
            audios = [a.squeeze(0) for a in audios]  # (1, T) -> (T, )
            audio_lens = torch.tensor([a.size(0) for a in audios], dtype=torch.int64)
            audios = collate_vectors(audios, padding_value=0)

            out = out + (audios, audio_lens)

        if self.fault_tolerant:
            out = out + (cuts,)

        return out

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
        start_frames, nums_frames = zip(
            *(
                supervision_to_frames(
                    sup, self.extractor.frame_shift, cut.sampling_rate
                )
                for cut in cuts
                for sup in cut.supervisions
            )
        )
        sequence_idx = [i for i, c in enumerate(cuts) for s in c.supervisions]
        return {
            "sequence_idx": torch.tensor(sequence_idx, dtype=torch.int32),
            "start_frame": torch.tensor(start_frames, dtype=torch.int32),
            "num_frames": torch.tensor(nums_frames, dtype=torch.int32),
        }

    def supervision_masks(
        self, cuts: CutSet, use_alignment_if_exists: Optional[str] = None
    ) -> torch.Tensor:
        """Returns the mask for supervised samples.

        :param use_alignment_if_exists: optional str, key for alignment type to use for generating the mask. If not
            exists, fall back on supervision time spans.
        """
        return collate_vectors(
            [
                compute_supervisions_frame_mask(
                    cut,
                    frame_shift=self.extractor.frame_shift,
                    use_alignment_if_exists=use_alignment_if_exists,
                )
                for cut in cuts
            ]
        )


@lru_cache(maxsize=1)
def _get_executor(
    max_workers: int = 0, executor_type: Type[ExecutorType] = ThreadPoolExecutor
) -> Optional[Executor]:
    """
    This function caches a thread/process pool in the global state of a given process.
    It's useful for keeping a process pool alive across different invocations within the
    same process for efficiency.
    We intend it to be used for efficient data reads withing a task executed in a
    parent process pool.
    """
    if max_workers <= 0:
        return None
    return executor_type(max_workers=max_workers)
