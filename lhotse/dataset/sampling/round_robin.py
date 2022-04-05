from functools import reduce
from operator import add
from typing import Any, Callable, Dict, Optional, Tuple, Union

from lhotse import CutSet
from lhotse.cut import Cut
from lhotse.dataset.sampling.base import CutSampler, SamplingDiagnostics


class RoundRobinSampler(CutSampler):
    """
    :class:`.RoundRobinSampler` takes several samplers as input, and yields
    a mini-batch of cuts from each of those samplers in turn.
    E.g., with two samplers, the first mini-batch is from ``sampler0``,
    the seconds from ``sampler1``, the third from ``sampler0``, and so on.
    It is helpful for alternating mini-batches from multiple datasets or manually
    creating batches of different sizes.

    The input samplers do not have to provide the same number of batches -- when
    any of the samplers becomes depleted, we continue to iterate the non-depleted
    samplers, until all of them are exhausted.

    Example::

        >>> sampler = RoundRobinSampler(
        ...     SimpleCutSampler(cuts_corpusA, max_cuts=32, shuffle=True),
        ...     SimpleCutSampler(cuts_corpusB, max_cuts=64, shuffle=True),
        ... )
        >>> for cut in sampler:
        ...     pass  # profit
    """

    def __init__(self, *samplers: CutSampler) -> None:
        """
        RoundRobinSampler's constructor.

        :param samplers: The list of samplers from which we sample batches in turns.
        """
        super().__init__(rank=0, world_size=1)
        self.samplers = samplers
        self._nondepleted_samplers_indices = list(range(len(self.samplers)))
        self._cur_sampler_idx = 0

    @property
    def remaining_duration(self) -> Optional[float]:
        """
        Remaining duration of data left in the sampler (may be inexact due to float arithmetic).
        """
        try:
            return sum(s.remaining_duration for s in self.samplers)
        except TypeError:
            return None

    @property
    def remaining_cuts(self) -> Optional[int]:
        """
        Remaining number of cuts in the sampler.
        Not available when the CutSet is read in lazy mode (returns None).
        """
        try:
            return sum(s.remaining_cuts for s in self.samplers)
        except TypeError:
            return None

    @property
    def num_cuts(self) -> Optional[int]:
        """
        Total number of cuts in the sampler.
        Not available when the CutSet is read in lazy mode (returns None).
        """
        try:
            return sum(s.num_cuts for s in self.samplers)
        except TypeError:
            return None

    def allow_iter_to_reset_state(self):
        """
        Enables re-setting to the start of an epoch when iter() is called.
        This is only needed in one specific scenario: when we restored previous
        sampler state via ``sampler.load_state_dict()`` but want to discard
        the progress in the current epoch and start from the beginning.
        """
        super().allow_iter_to_reset_state()
        for s in self.samplers:
            s.allow_iter_to_reset_state()

    def state_dict(self) -> Dict[str, Any]:
        """
        Return the current state of the sampler in a state_dict.
        Together with ``load_state_dict()``, this can be used to restore the
        training loop's state to the one stored in the state_dict.
        """
        state_dict = super().state_dict()
        state_dict.update(
            {
                "samplers": [s.state_dict() for s in self.samplers],
                "_cur_sampler_idx": self._cur_sampler_idx,
                # Explicit list copy below allows to restore within the same process.
                "_nondepleted_samplers_indices": list(
                    self._nondepleted_samplers_indices
                ),
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
        self._cur_sampler_idx = state_dict.pop("_cur_sampler_idx")
        self._nondepleted_samplers_indices = state_dict.pop(
            "_nondepleted_samplers_indices"
        )
        assert len(self.samplers) == len(state_dict["samplers"]), (
            "Error in RoundRobinSampler.load_state_dict(): Inconsistent number of samplers: "
            f"current RoundRobinSampler has {len(self.samplers)}, the state_dict has {len(state_dict['samplers'])}."
        )
        for sampler, sampler_sd in zip(self.samplers, state_dict.pop("samplers")):
            sampler.load_state_dict(sampler_sd)
        super().load_state_dict(state_dict)

    def __iter__(self):
        for sampler in self.samplers:
            iter(sampler)
        if self._just_restored_state:
            return self
        self._nondepleted_samplers_indices = list(range(len(self.samplers)))
        self._cur_sampler_idx = 0
        return self

    def _next_batch(self) -> Union[CutSet, Tuple[CutSet]]:
        if len(self._nondepleted_samplers_indices) == 0:
            raise StopIteration()

        sampler_idx = self._nondepleted_samplers_indices[self._cur_sampler_idx]
        sampler = self.samplers[sampler_idx]

        try:
            batch = next(sampler)
        except StopIteration:
            # Try again recursively as long as there is at least one non depleted sampler left.
            self._nondepleted_samplers_indices.pop(self._cur_sampler_idx)
            return self._next_batch()

        self._cur_sampler_idx = (self._cur_sampler_idx + 1) % len(
            self._nondepleted_samplers_indices
        )

        return batch

    def set_epoch(self, epoch: int) -> None:
        """
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        :param epoch: Epoch number.
        """
        for s in self.samplers:
            s.set_epoch(epoch)
        super().set_epoch(epoch)

    def filter(self, predicate: Callable[[Cut], bool]) -> None:
        """
        Add a constraint on individual cuts that has to be satisfied to consider them.

        Can be useful when handling large, lazy manifests where it is not feasible to
        pre-filter them before instantiating the sampler.

        Example:
            >>> cuts = CutSet(...)
            ... sampler = SimpleCutSampler(cuts, max_duration=100.0)
            ... # Retain only the cuts that have at least 1s and at most 20s duration.
            ... sampler.filter(lambda cut: 1.0 <= cut.duration <= 20.0)
        """
        for sampler in self.samplers:
            sampler.filter(predicate)

    @property
    def diagnostics(self) -> SamplingDiagnostics:
        return reduce(add, (s.diagnostics for s in self.samplers))

    def get_report(self) -> str:
        """Returns a string describing the statistics of the sampling process so far."""
        return self.diagnostics.get_report()
