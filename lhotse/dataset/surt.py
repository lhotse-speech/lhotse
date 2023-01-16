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
            'supervisions': [
                {
                    'sequence_idx': Tensor[int] of shape (S,)
                    'text': List[str] of len S

                    # For feature input strategies
                    'start_frame': Tensor[int] of shape (S,)
                    'num_frames': Tensor[int] of shape (S,)

                    # For audio input strategies
                    'start_sample': Tensor[int] of shape (S,)
                    'num_samples': Tensor[int] of shape (S,)

                    # Optionally, when return_cuts=True
                    'cut': List[AnyCut] of len S
                }
            ]
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
        self.num_channels = num_channels
        self.text_delimiter = text_delimiter
        self.cut_transforms = ifnone(cut_transforms, [])
        self.input_transforms = ifnone(input_transforms, [])
        self.input_strategy = input_strategy

        # This attribute is a workaround to constantly growing HDF5 memory
        # throughout the epoch. It regularly closes open file handles to
        # reset the internal HDF5 caches.
        self.hdf5_fix = Hdf5MemoryIssueFix(reset_interval=100)

        assert num_channels == 2, "Only 2 channels are supported for now."

    def __getitem__(self, cuts: CutSet) -> Dict[str, Union[torch.Tensor, List[str]]]:
        """
        Return a new batch, with the batch size automatically determined using the constraints
        of max_frames and max_cuts.
        """
        validate_for_asr(cuts)

        self.hdf5_fix.update()

        # Sort the cuts by duration so that the first one determines the batch time dimensions.
        cuts = cuts.sort_by_duration(ascending=False)

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
        supervisions_ch0 = defaultdict(list)
        supervisions_ch1 = defaultdict(list)
        for cut in cuts:
            supervisions_ch0[cut.id] = []
            supervisions_ch1[cut.id] = []
            for sup in cut.supervisions:
                if (
                    len(supervisions_ch0[cut.id]) == 0
                    or supervisions_ch0[cut.id][-1].end < sup.start
                ):
                    supervisions_ch0[cut.id].append(sup)
                else:
                    supervisions_ch1[cut.id].append(sup)

        batch = {
            "inputs": inputs,
            "input_lens": input_lens,
            "supervisions_ch0": list(supervisions_ch0.values()),
            "supervisions_ch1": list(supervisions_ch1.values()),
            "text_ch0": default_collate(
                [
                    self.text_delimiter.join([sup.text for sup in cut_sups])
                    for cut_sups in supervisions_ch0.values()
                ]
            ),
            "text_ch1": default_collate(
                [
                    self.text_delimiter.join([sup.text for sup in cut_sups])
                    for cut_sups in supervisions_ch1.values()
                ]
            ),
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
