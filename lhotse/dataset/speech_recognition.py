from lhotse.dataset.core import SpeechDataset
from lhotse.dataset import fields


class K2SpeechRecognitionIterableDataset(SpeechDataset):
    """
    The PyTorch Dataset for the speech recognition task using K2 library.

    This dataset internally batches and collates the Cuts and should be used with
    PyTorch DataLoader with argument batch_size=None to work properly.
    The batch size is determined automatically to satisfy the constraints of ``max_frames``
    and ``max_cuts``.

    This dataset will automatically partition itself when used with a multiprocessing DataLoader
    (i.e. the same cut will not appear twice in the same epoch).

    By default, we "pack" the batches to minimize the amount of padding - we achieve that by
    concatenating the cuts' feature matrices with a small amount of silence (padding) in between.

    Each item in this dataset is a dict of:

    .. code-block::

        {
            'features': float tensor of shape (B, T, F)
            'supervisions': [
                {
                    'cut_id': List[str] of len S
                    'sequence_idx': Tensor[int] of shape (S,)
                    'text': List[str] of len S
                    'start_frame': Tensor[int] of shape (S,)
                    'num_frames': Tensor[int] of shape (S,)
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
            self, *args, concat_cuts=True, **kwargs
    ):
        """
        K2 ASR IterableDataset constructor.

        :param cuts: the ``CutSet`` to sample data from.
        :param max_frames: The maximum number of feature frames that we're going to put in a single batch.
            The padding frames do not contribute to that limit, since we pack the batch by default to minimze
            the amount of padding.
        :param max_cuts: The maximum number of cuts sampled to form a mini-batch.
            By default, this constraint is off.
        :param shuffle: When ``True``, the cuts will be shuffled at the start of iteration.
            Convenient when mini-batch loop is inside an outer epoch-level loop, e.g.:
            `for epoch in range(10): for batch in dataset: ...` as every epoch will see a
            different cuts order.
        :param return_cuts: When ``True``, will additionally return a "cut" field in each batch with the Cut
            objects used to create that batch.
        :param concat_cuts: When ``True``, we will concatenate the cuts to minimize the total amount of padding;
            e.g. instead of creating a batch with 40 examples, we will merge some of the examples together
            adding some silence between them to avoid a large number of padding frames that waste the computation.
            Enabled by default.
        :param concat_cuts_gap: The duration of silence in seconds that is inserted between the cuts;
            it's goal is to let the model "know" that there are separate utterances in a single example.
        :param concat_cuts_duration_factor: Determines the maximum duration of the concatenated cuts;
            by default it's 1, setting the limit at the duration of the longest cut in the batch.
        """
        super().__init__(
            *args,
            signal_fields=[fields.Feats()],
            supervision_fields=[fields.Text(), fields.FeatureSpan()],
            concat_cuts=concat_cuts,
            **kwargs
        )
