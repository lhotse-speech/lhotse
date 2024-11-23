import warnings
from typing import Any, Dict, List, Optional

from lhotse import CutSet, Seconds
from lhotse.dataset.sampling.base import TimeConstraint
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

    def state_dict(self) -> Dict[str, Any]:
        """
        Return the current state of the sampler in a state_dict.
        Together with ``load_state_dict()``, this can be used to restore the
        training loop's state to the one stored in the state_dict.
        """
        state_dict = super().state_dict()
        state_dict.update(
            {
                "time_constraint": self.time_constraint.state_dict(),
                "weights": self.weights,
                "num_samples": self.num_samples,
            }
        )
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Restore the state of the sampler that is described in a state_dict.
        This will result in the sampler yielding batches from where the previous training left it off.

        .. caution::
            The samplers are expected to be initialized with the same CutSets,
            but this is not explicitly checked anywhere.

        .. caution::
            The input ``state_dict`` is being mutated: we remove each consumed key, and expect
            it to be empty at the end of loading. If you don't want this behavior, pass a copy
            inside of this function (e.g., using ``import deepcopy``).

        .. note::
            For implementers of sub-classes of CutSampler: the flag ``self._just_restored_state`` has to be
            handled in ``__iter__`` to make it avoid resetting the just-restored state (only once).
        """
        time_constraint = TimeConstraint(**state_dict.pop("time_constraint"))
        if self.time_constraint != time_constraint:
            warnings.warn(
                "SimpleCutSampler.load_state_dict(): Inconsistent time_constraint:\n"
                f"expected {self.time_constraint}\n"
                f"received {time_constraint}\n"
                f"We will overwrite the settings with the received state_dict."
            )
        self.time_constraint = time_constraint

        super().load_state_dict(state_dict)

        # Restore the data source's state
        self.data_source.fast_forward(self.diagnostics.current_epoch_stats.total_cuts)

        self.weights = state_dict.pop("weights")
        self.num_samples = state_dict.pop("num_samples")

    def __iter__(self) -> "WeightedSimpleCutSampler":
        """
        Prepare the dataset for iterating over a new epoch. Will shuffle the data if requested.
        """
        # Restored state with load_state_dict()? Skip resetting only this once.
        if self._just_restored_state:
            return self
        # Why reset the current epoch?
        # Either we are iterating the epoch for the first time and it's a no-op,
        # or we are iterating the same epoch again, in which case setting more steps
        # than are actually available per epoch would have broken the checkpoint restoration.
        self.diagnostics.reset_current_epoch()
        # Reset the state to the beginning of the epoch.
        iter(self.data_source)
        return self
