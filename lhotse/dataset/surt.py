from collections import defaultdict
from typing import Callable, Dict, List, Union

import numpy as np
import torch
import torch.nn.functional as F

from lhotse import validate
from lhotse.cut import CutSet
from lhotse.dataset.input_strategies import BatchIO, PrecomputedFeatures
from lhotse.utils import LOG_EPSILON, compute_num_frames, ifnone
from lhotse.workarounds import Hdf5MemoryIssueFix


class K2SurtDataset(torch.utils.data.Dataset):
    """
    The PyTorch Dataset for the multi-talker ASR task using k2 library.
    We support the modeling framework known as Streaming Unmixing and Recognition
    Transducer (SURT), as described in [1] and [2], but this dataset can also be
    used for other multi-talker ASR approaches such as MT-RNNT [3] and SOT [4].
    See icefall recipe for usage: https://github.com/k2-fsa/icefall/pull/1126.

    We take a cut containing possibly overlapping speech and split the supervision
    segments into one of N channels based on their start times (N is provided), known
    as ``heuristic error assignment training`` (HEAT) [1]. The supervision segments
    in each channel are then concatenated and used as the supervision for that channel.
    If we have features for the source cuts, we can also return them for use in masking
    losses, for instance.

    [1] Lu, L., Kanda, N., Li, J., & Gong, Y. (2021). Streaming end-to-end multi-talker
    speech recognition. IEEE Signal Processing Letters, 28, 803-807.

    [2] Raj, D., Lu, L., Chen, Z., Gaur, Y., & Li, J. (2022, May). Continuous streaming
    multi-talker asr with dual-path transducers. In ICASSP 2022-2022 IEEE International
    Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 7317-7321). IEEE.

    [3] Sklyar, I., Piunova, A., Zheng, X., & Liu, Y. (2022, May). Multi-turn RNN-T for
    streaming recognition of multi-party speech. In ICASSP 2022-2022 IEEE International
    Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 8402-8406). IEEE.

    [4] Kanda, N., Gaur, Y., Wang, X., Meng, Z., & Yoshioka, T. (2020). Serialized output
    training for end-to-end overlapped speech recognition. arXiv preprint arXiv:2003.12687.

    .. hint:: Training mixtures can be simulated from single-speaker utterances using
        :class:`~lhotse.workflows.MeetingSimulation` workflow.

    This dataset expects to be queried with lists of cut IDs,
    for which it loads features and automatically collates/batches them.

    To use it with a PyTorch DataLoader, set ``batch_size=None``
    and provide a :class:`SimpleCutSampler` sampler.

    Each item in this dataset is a dict of:

    .. code-block::

        {
            'inputs': float tensor with shape determined by :attr:`input_strategy`:
                      - single-channel:
                        - features: (B, T, F)
                        - audio: (B, T)
                      - multi-channel: currently not supported
            'input_lens': int tensor of shape (B,)
            'supervisions': list of lists of supervision segments, where the outer list is
                        batch, and the inner list is indexed by channel. So ``len(supervisions) == B``,
                        and ``len(supervisions[i]) == num_channels``. Note that some channels may
                        have no supervision segments.
            'text': list of lists of strings, where the outer list is batch, and the inner list
                    is indexed by channel. So ``len(text) == B``, and ``len(text[i]) == num_channels``.
                    Each element contains the text of the supervision segments in that channel,
                    joined by the :attr:`text_delimiter`. Note that some channels may have no
                    supervision segments, so the corresponding text will be an empty string.
        }

    If ``return_cuts`` is ``True``, each item will also contain a ``cuts`` field with the
    :class:`~lhotse.cut.Cut` objects used to create the batch.

    Additionally, if ``return_sources`` is ``True``, each item will contain:

    .. code-block::

        {
            'source_feats': list of list of float tensors. The outer list is batch, and the inner
                            list is number of segments in the mixture. Each element denotes
                            the features of the source cut in the mixture.
            'source_boundaries': list of list of tuples, where the outer list is batch, and the inner list
                                is number of segments in the mixture. Each element denotes
                                the start and end frame of the source cut in the mixture.
        }

    In order to return the source features and boundaries, we expect the cuts to contain
    some additional fields:

    - ``source_feats``: a float tensor representing the features of all the source segments,
        concatenated together along T dimension, in order of their start times.
    - ``source_feat_offsets``: a list of ints, where each element denotes the offset of the
        source segments in the ``source_feats`` tensor.

    See https://github.com/lhotse-speech/lhotse/discussions/1008#discussioncomment-5511746
    for example code to create these fields.


    Dimension symbols legend:
    * ``B`` - batch size (number of Cuts)
    * ``S`` - number of supervision segments (greater or equal to B, as each Cut may have multiple supervisions)
    * ``T`` - number of frames of the longest Cut
    * ``F`` - number of features

    The 'sequence_idx' field is the index of the Cut used to create the example in the Dataset.
    """

    def __init__(
        self,
        return_cuts: bool = False,
        return_sources: bool = False,
        return_alignments: bool = False,
        num_channels: int = 2,
        text_delimiter: str = " ",
        cut_transforms: List[Callable[[CutSet], CutSet]] = None,
        input_transforms: List[Callable[[torch.Tensor], torch.Tensor]] = None,
        input_strategy: BatchIO = PrecomputedFeatures(),
        pad_value: float = LOG_EPSILON,
        strict: bool = False,
    ):
        """
        k2 ASR IterableDataset constructor.

        :param return_cuts: When ``True``, will additionally return a "cut" field in each batch with the Cut
            objects used to create that batch.
        :param return_sources: When ``True``, will additionally return a "source_feats" field and a "source_boundaries"
            field in each batch. The "source_feats" field contains the features of the source cuts from
            which the mixture was created, and "source_boundaries" contains the boundaries of the source cuts
            in the mixture (in number of frames). This requires that the cuts contain additional fields
            ``source_feats`` (which is a TemporalArray) and ``source_feat_offsets`` (which is a list of ints).
        :param return_alignments: When ``True``, will keep the supervision alignments if they
            are present in the cuts.
        :param num_channels: Number of output branches. The supervision utterances will be
            split into the channels based on their start times.
        :param text_delimiter: The delimiter used to join the text of the supervision segments in
            each channel.
        :param cut_transforms: A list of transforms to be applied on each sampled batch,
            before converting cuts to an input representation (audio/features).
            Examples: cut concatenation, noise cuts mixing, etc.
        :param input_transforms: A list of transforms to be applied on each sampled batch,
            after the cuts are converted to audio/features.
            Examples: normalization, SpecAugment, etc.
        :param input_strategy: The strategy used to convert the cuts to audio/features.
        :param pad_value: The value used to pad the source features to resolve one-off errors.
        :param strict: If ``True``, we will remove cuts that have more simultaneous supervisions
            than the number of channels. If ``False``, we will keep them.
        """
        super().__init__()
        # Initialize the fields
        self.return_cuts = return_cuts
        self.return_sources = return_sources
        self.return_alignments = return_alignments
        self.num_channels = num_channels
        self.text_delimiter = text_delimiter
        self.cut_transforms = ifnone(cut_transforms, [])
        self.input_transforms = ifnone(input_transforms, [])
        self.input_strategy = input_strategy
        self.pad_value = pad_value
        self.strict = strict

        # This attribute is a workaround to constantly growing HDF5 memory
        # throughout the epoch. It regularly closes open file handles to
        # reset the internal HDF5 caches.
        self.hdf5_fix = Hdf5MemoryIssueFix(reset_interval=100)

    def __getitem__(self, cuts: CutSet) -> Dict[str, Union[torch.Tensor, List[str]]]:
        """
        Return a new batch, with the batch size automatically determined using the constraints
        of max_duration and max_cuts.
        """
        validate_for_asr(cuts)

        self.hdf5_fix.update()

        if not self.return_alignments:
            cuts = cuts.drop_alignments()

        # Sort the cuts by duration so that the first one determines the batch time dimensions.
        cuts = cuts.sort_by_duration(ascending=False)

        # Optional CutSet transforms - e.g. padding, or speed perturbation that adjusts
        # the supervision boundaries.
        for tnfm in self.cut_transforms:
            cuts = tnfm(cuts)

        # Assign supervisions to channels based on their start times.
        # ``supervisions`` is a dict indexed by cut id, and each value is a list of
        # lists of supervisions. The outer list is indexed by channel, and the inner
        # list contains the supervisions for that channel.
        supervisions = defaultdict(list)
        invalid_cuts = []
        source_feats = []
        source_boundaries = []

        for cut in cuts:
            cut_sups = [[] for _ in range(self.num_channels)]
            last_sup_end = [0.0 for _ in range(self.num_channels)]

            cut_sources = []
            cut_source_boundaries = []
            invalid_cut = False

            for sup in sorted(cut.supervisions, key=lambda s: s.start):
                # Assign the supervision to the first channel that is either empty or
                # has a supervision that ends before the current supervision starts.
                assigned = False
                for i in range(self.num_channels):
                    if len(cut_sups[i]) == 0 or last_sup_end[i] <= sup.start:
                        cut_sups[i].append(sup)
                        last_sup_end[i] = max(last_sup_end[i], sup.end)
                        assigned = True
                        break

                if not assigned:
                    # If we reach here, it means that there is no channel that is empty
                    # or has a supervision that ends before the current supervision starts.
                    # This is possible if number of overlapping speakers is more than the
                    # number of available channels. In this case, we assign the supervision
                    # so as to minimize the overlapping part. For this, we select the
                    # channel which ends the earliest.
                    invalid_cut = True
                    min_end_channel = last_sup_end.index(min(last_sup_end))
                    cut_sups[min_end_channel].append(sup)
                    last_sup_end[min_end_channel] = max(
                        last_sup_end[min_end_channel], sup.end
                    )

            if self.return_sources:
                source_feat_offsets = cut.source_feat_offsets
                assert len(source_feat_offsets) == len(cut.supervisions), (
                    "The number of source feature offsets should be equal to the number of supervisions."
                    f"Got {len(source_feat_offsets)} offsets for {len(cut.supervisions)} supervisions."
                )
                # To get cut_sources, we split the source_feats into a list of tensors
                # based on the source_feat_offsets.
                cut_sources = [
                    torch.from_numpy(x)
                    for x in np.split(cut.load_source_feats(), source_feat_offsets[1:])
                ]
                # To get cut_source_boundaries, we create (start, end) tuples based on
                # the supervision start and end times.
                cut_source_boundaries = [
                    (
                        compute_num_frames(
                            sup.start, cut.frame_shift, cut.sampling_rate
                        ),
                        compute_num_frames(sup.end, cut.frame_shift, cut.sampling_rate),
                    )
                    for sup in sorted(
                        cut.supervisions, key=lambda s: (s.start, s.speaker)
                    )
                ]
                # Adjust the source feats to fix one-off errors in the source_feat_offsets.
                cut_sources = [
                    adjust_source_feats(x, end - start, padding_value=self.pad_value)
                    for x, (start, end) in zip(cut_sources, cut_source_boundaries)
                ]

            if invalid_cut and self.strict:
                invalid_cuts.append(cut.id)
                continue
            supervisions[cut.id] = cut_sups
            if self.return_sources:
                source_feats.append(cut_sources)
                source_boundaries.append(cut_source_boundaries)

        # Remove invalid cuts.
        if len(invalid_cuts) > 0:
            print(
                f"WARNING: {len(invalid_cuts)} cuts were removed out of {len(cuts)} due to more overlapping speakers than channels."
            )
            cuts = cuts.filter(lambda cut: cut.id not in invalid_cuts)

        # Get a tensor with batched feature matrices, shape (B, T, F)
        # Collation performs auto-padding, if necessary.
        input_tpl = self.input_strategy(cuts)
        if len(input_tpl) == 3:
            # An input strategy with fault tolerant audio reading mode.
            # "cuts" may be a subset of the original "cuts" variable,
            # that only has cuts for which we succesfully read the audio.
            inputs, input_lens, cuts = input_tpl
        else:
            inputs, input_lens = input_tpl

        batch = {
            "inputs": inputs,
            "input_lens": input_lens,
            "supervisions": list(supervisions.values()),
            "text": [
                [
                    self.text_delimiter.join([sup.text.strip() for sup in sups_ch])
                    for sups_ch in cut_sups
                ]
                for cut_sups in supervisions.values()
            ],
        }
        if self.return_cuts:
            batch["cuts"] = cuts
        if self.return_sources:
            batch["source_feats"] = source_feats
            batch["source_boundaries"] = source_boundaries
        return batch


def adjust_source_feats(feats, num_frames, padding_value=0.0, tol=2):
    """
    Adjust the number of frames in the source features to match the supervision.
    If the source features have fewer frames than the supervision, we pad them
    to match the supervision. If the source features have more frames than the
    supervision, we trim them to match the supervision.

    Args:
        feats: Source features.
        num_frames: Number of frames in the supervision.
        padding_value: Value to use for padding.
        tol: Tolerance for checking if the number of frames in the source features
            is close to the number of frames in the supervision.
    """
    if feats.shape[0] == num_frames:
        return feats
    elif abs(feats.shape[0] - num_frames) > tol:
        raise ValueError(
            f"Number of frames in the source features ({feats.shape[0]}) is not close to "
            f"the number of frames in the supervision ({num_frames})."
        )
    elif feats.shape[0] < num_frames:
        # If the source features have fewer frames than the supervision,
        # we pad them to match the supervision.
        return F.pad(feats, (0, 0, 0, num_frames - feats.shape[0]), value=padding_value)
    else:
        # If the source features have more frames than the supervision,
        # we trim them to match the supervision.
        return feats[:num_frames]


def validate_for_asr(cuts: CutSet) -> None:
    validate(cuts)
    tol = 2e-3  # 1ms
    for cut in cuts:
        for supervision in cut.supervisions:
            assert supervision.start >= -tol, (
                f"Supervisions starting before the cut are not supported for ASR"
                f" (sup id: {supervision.id}, cut id: {cut.id})"
            )
            assert supervision.duration <= cut.duration + tol, (
                f"Supervisions ending after the cut "
                f"are not supported for ASR"
                f" (sup id: {supervision.id}, cut id: {cut.id})"
            )
