from collections import defaultdict
from typing import Callable, Dict, List, Union

import torch
from torch.utils.data.dataloader import default_collate

from lhotse import validate
from lhotse.cut import CutSet
from lhotse.dataset.input_strategies import BatchIO, PrecomputedFeatures
from lhotse.utils import ifnone
from lhotse.workarounds import Hdf5MemoryIssueFix


class K2SurtDataset(torch.utils.data.Dataset):
    """
    The PyTorch Dataset for the multi-talker SURT task using k2 library.
    See icefall recipe for usage.

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
        return_alignments: bool = False,
        num_channels: int = 2,
        text_delimiter: str = " ",
        cut_transforms: List[Callable[[CutSet], CutSet]] = None,
        input_transforms: List[Callable[[torch.Tensor], torch.Tensor]] = None,
        input_strategy: BatchIO = PrecomputedFeatures(),
    ):
        """
        k2 ASR IterableDataset constructor.

        :param return_cuts: When ``True``, will additionally return a "cut" field in each batch with the Cut
            objects used to create that batch.
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
        """
        super().__init__()
        # Initialize the fields
        self.return_cuts = return_cuts
        self.return_alignments = return_alignments
        self.num_channels = num_channels
        self.text_delimiter = text_delimiter
        self.cut_transforms = ifnone(cut_transforms, [])
        self.input_transforms = ifnone(input_transforms, [])
        self.input_strategy = input_strategy

        # This attribute is a workaround to constantly growing HDF5 memory
        # throughout the epoch. It regularly closes open file handles to
        # reset the internal HDF5 caches.
        self.hdf5_fix = Hdf5MemoryIssueFix(reset_interval=100)

    def __getitem__(self, cuts: CutSet) -> Dict[str, Union[torch.Tensor, List[str]]]:
        """
        Return a new batch, with the batch size automatically determined using the constraints
        of max_frames and max_cuts.
        """
        validate_for_asr(cuts)

        self.hdf5_fix.update()

        # Sort the cuts by duration so that the first one determines the batch time dimensions.
        cuts = cuts.sort_by_duration(ascending=False)
        if not self.return_alignments:
            cuts = cuts.drop_alignments()

        # Optional CutSet transforms - e.g. padding, or speed perturbation that adjusts
        # the supervision boundaries.
        for tnfm in self.cut_transforms:
            cuts = tnfm(cuts)

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

        # Assign supervisions to channels based on their start times.
        # ``supervisions`` is a dict indexed by cut id, and each value is a list of
        # lists of supervisions. The outer list is indexed by channel, and the inner
        # list contains the supervisions for that channel.
        supervisions = defaultdict(list)
        for cut in cuts:
            cut_sups = [[] for _ in range(self.num_channels)]
            last_sup_end = [0.0 for _ in range(self.num_channels)]

            for sup in sorted(cut.supervisions, key=lambda s: s.start):
                # Assign the supervision to the first channel that is either empty or
                # has a supervision that ends before the current supervision starts.
                assigned = False
                for i in range(self.num_channels):
                    if len(cut_sups[i]) == 0 or cut_sups[i][-1].end < sup.start:
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
                    min_end_channel = last_sup_end.index(min(last_sup_end))
                    cut_sups[min_end_channel].append(sup)

            supervisions[cut.id] = cut_sups

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
        return batch


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
