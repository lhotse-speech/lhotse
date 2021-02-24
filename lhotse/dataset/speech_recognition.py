from typing import Callable, Dict, List, Union

import torch
from torch.utils.data.dataloader import DataLoader, default_collate

from lhotse import validate
from lhotse.cut import CutSet
from lhotse.dataset.collation import collate_features
from lhotse.utils import supervision_to_frames


class K2SpeechRecognitionDataset(torch.utils.data.Dataset):
    """
    The PyTorch Dataset for the speech recognition task using K2 library.

    This dataset expects to be queried with lists of cut IDs,
    for which it loads features and automatically collates/batches them.

    To use it with a PyTorch DataLoader, set ``batch_size=None``
    and provide a :class:`SingleCutSampler` sampler.

    Each item in this dataset is a dict of:

    .. code-block::

        {
            'features': float tensor of shape (B, T, F)
            'supervisions': [
                {
                    'sequence_idx': Tensor[int] of shape (S,)
                    'text': List[str] of len S
                    'start_frame': Tensor[int] of shape (S,)
                    'num_frames': Tensor[int] of shape (S,)
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
            cuts: CutSet,
            return_cuts: bool = False,
            cut_transforms: List[Callable[[CutSet], CutSet]] = None
    ):
        """
        K2 ASR IterableDataset constructor.

        :param cuts: the ``CutSet`` to sample data from.
        :param return_cuts: When ``True``, will additionally return a "cut" field in each batch with the Cut
            objects used to create that batch.
        :param cut_transforms: A list of transforms to be applied on each sampled batch
            (e.g. cut concatenation, noise cuts mixing, etc.).
        """
        super().__init__()
        # Initialize the fields
        self.cuts = cuts
        self.return_cuts = return_cuts
        self.cut_transforms = cut_transforms if cut_transforms is not None else ()
        self._validate()

    def __len__(self) -> int:
        return len(self.cuts)

    def __getitem__(self, cut_ids: List[str]) -> Dict[str, Union[torch.Tensor, List[str]]]:
        """
        Return a new batch, with the batch size automatically determined using the contraints
        of max_frames and max_cuts.
        """
        # Collect the cuts that will form a batch, satisfying the criteria of max_cuts and max_frames.
        # The returned object is a CutSet that we can keep on modifying (e.g. padding, mixing, etc.)
        cuts = self.cuts.subset(cut_ids=cut_ids)

        # Sort the cuts by duration so that the first one determines the batch time dimensions.
        cuts = cuts.sort_by_duration(ascending=False)

        # Optional transforms.
        for tnfm in self.cut_transforms:
            cuts = tnfm(cuts)

        # Get a tensor with batched feature matrices, shape (B, T, F)
        # Collation performs auto-padding, if necessary.
        features = collate_features(cuts)

        batch = {
            'features': features,
            'supervisions': default_collate([
                {
                    'sequence_idx': sequence_idx,
                    'text': supervision.text,
                    'start_frame': start_frame,
                    'num_frames': num_frames
                }
                for sequence_idx, cut in enumerate(cuts)
                for supervision, (start_frame, num_frames) in zip(
                    cut.supervisions,
                    (
                        supervision_to_frames(s, cut.frame_shift, cut.sampling_rate, max_frames=cut.num_frames)
                        for s in cut.supervisions
                    )
                )
            ])
        }
        if self.return_cuts:
            batch['supervisions']['cut'] = [cut for cut in cuts for sup in cut.supervisions]

        return batch

    def _validate(self) -> None:
        validate(self.cuts)
        tol = 1e-3  # 1ms
        for cut in self.cuts:
            for supervision in cut.supervisions:
                assert supervision.start >= -tol, f"Supervisions starting before the cut are not supported for ASR" \
                                                  f" (sup id: {supervision.id}, cut id: {cut.id})"
                assert supervision.duration <= cut.duration + tol, f"Supervisions ending after the cut " \
                                                                   f"are not supported for ASR" \
                                                                   f" (sup id: {supervision.id}, cut id: {cut.id})"
