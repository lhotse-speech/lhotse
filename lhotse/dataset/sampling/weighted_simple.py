from typing import List, Optional

from lhotse import CutSet, Seconds
from lhotse.dataset.sampling.data_source import WeightedDataSource
from lhotse.dataset.sampling.simple import SimpleCutSampler


class WeightedSimpleCutSampler(SimpleCutSampler):
    """
    Samples cuts from a CutSet, where the sampling prob is given by a list.
    To enable global sampling, cuts must be in eager mode.

    When performing sampling, it avoids having duplicated cuts in the same batch.
    The sampler terminates if the number of sampled cuts reach :attr:`num_samples`

    When one of :attr:`max_duration`, or :attr:`max_cuts` is specified,
    the batch size is dynamic.

    Example usage:

        >>> dataset = K2SpeechRecognitionDataset(cuts)
        >>> weights = get_weights(cuts)
        >>> sampler = WeightedSimpleCutSampler(cuts, weights, num_samples=100, max_duration=200.0)
        >>> loader = DataLoader(dataset, sampler=sampler, batch_size=None)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(
        self,
        cuts: CutSet,
        cuts_weight: List,
        num_samples: int,
        max_duration: Seconds = None,
        max_cuts: Optional[int] = None,
        shuffle: bool = False,
        drop_last: bool = False,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
    ):
        """
        WeightedSimpleCutSampler's constructor

        :param cuts: the ``CutSet`` to sample data from.
        :param cuts_weight: the weight of each cut for sampling.
        :param num_samples: the number of samples to be drawn.
        :param max_duration: The maximum total recording duration from ``cuts``.
        :param max_cuts: The maximum number of cuts sampled to form a mini-batch.
            By default, this constraint is off.
        :param shuffle: When ``True``, the cuts will be shuffled at the start of iteration.
            Convenient when mini-batch loop is inside an outer epoch-level loop, e.g.:
            `for epoch in range(10): for batch in dataset: ...` as every epoch will see a
            different cuts order.
        :param drop_last: When ``True``, the last batch is dropped if it's incomplete.
        :param world_size: Total number of distributed nodes. We will try to infer it by default.
        :param rank: Index of distributed node. We will try to infer it by default.
        :param seed: Random seed used to consistently shuffle the dataset across different processes.
        """
        super().__init__(
            cuts=cuts,
            drop_last=drop_last,
            shuffle=shuffle,
            world_size=world_size,
            rank=rank,
            max_duration=max_duration,
            max_cuts=max_cuts,
            seed=seed,
        )
        assert not cuts.is_lazy, "This sampler does not support lazy mode!"
        self.data_source = WeightedDataSource(
            cuts, weights=cuts_weight, num_samples=num_samples
        )

        self.weights = cuts_weight
        self.num_samples = num_samples

    def __iter__(self) -> "WeightedSimpleCutSampler":
        """
        Prepare the dataset for iterating over a new epoch. Will shuffle the data if requested.
        """
        # Why reset the current epoch?
        # Either we are iterating the epoch for the first time and it's a no-op,
        # or we are iterating the same epoch again, in which case setting more steps
        # than are actually available per epoch would have broken the checkpoint restoration.
        self.diagnostics.reset_current_epoch()
        # Reset the state to the beginning of the epoch.
        iter(self.data_source)
        return self
