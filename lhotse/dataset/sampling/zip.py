from functools import reduce
from operator import add
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from lhotse import CutSet
from lhotse.cut import Cut
from lhotse.dataset.sampling.base import CutSampler, SamplingDiagnostics


class ZipSampler(CutSampler):
    """
    :class:`.ZipSampler` takes several samplers as input and concatenates their
    sampled mini-batch cuts together into a single :class:`~lhotse.cut.CutSet`,
    or returns a tuple of the mini-batch CutSets.
    It is helpful for ensuring that each batch consists of some proportion of cuts
    coming from different sources.

    The input samplers do not have to provide the same number of batches -- when
    any of the samplers becomes depleted, the iteration will stop (like with
    Python's ``zip()`` function).

    Example::

        >>> sampler = ZipSampler(
        ...     SimpleCutSampler(cuts_corpusA, max_duration=250, shuffle=True),
        ...     SimpleCutSampler(cuts_corpusB, max_duration=100, shuffle=True),
        ... )
        >>> for cut in sampler:
        ...     pass  # profit
    """

    def __init__(self, *samplers: CutSampler, merge_batches: bool = True) -> None:
        """
        ZipSampler's constructor.

        :param samplers: The list of samplers from which we sample batches together.
        :param merge_batches: Should we merge the batches from each sampler into a single CutSet,
            or return a tuple of CutSets. Setting this to ``False`` makes ZipSampler behave
            more like Python's ``zip`` function.
        """
        super().__init__(rank=0, world_size=1)
        self.samplers = samplers
        self.merge_batches = merge_batches

    @property
    def remaining_duration(self) -> Optional[float]:
        """
        Remaining duration of data left in the sampler (may be inexact due to float arithmetic).

        .. note: For ZipSampler, it's the minimum of remaining durations in its sub-samplers.
        """
        try:
            return min(s.remaining_duration for s in self.samplers)
        except TypeError:
            return None

    @property
    def remaining_cuts(self) -> Optional[int]:
        """
        Remaining number of cuts in the sampler.
        Not available when the CutSet is read in lazy mode (returns None).

        .. note: For ZipSampler, it's the minimum of remaining cuts in its sub-samplers.
        """
        try:
            return min(s.remaining_cuts for s in self.samplers)
        except TypeError:
            return None

    @property
    def num_cuts(self) -> Optional[int]:
        """
        Total number of cuts in the sampler.
        Not available when the CutSet is read in lazy mode (returns None).

        .. note: For ZipSampler, it's the minimum of num cuts in its sub-samplers.
        """
        try:
            return min(s.num_cuts for s in self.samplers)
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
                "merge_batches": self.merge_batches,
                "samplers": [s.state_dict() for s in self.samplers],
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
        self.merge_batches = state_dict.pop("merge_batches")
        assert len(self.samplers) == len(state_dict["samplers"]), (
            "Error in ZipSampler.load_state_dict(): Inconsistent number of samplers: "
            f"current ZipSampler has {len(self.samplers)}, the state_dict has {len(state_dict['samplers'])}."
        )
        for sampler, sampler_sd in zip(self.samplers, state_dict.pop("samplers")):
            sampler.load_state_dict(sampler_sd)
        super().load_state_dict(state_dict)

    def __iter__(self):
        for sampler in self.samplers:
            iter(sampler)
        return self

    def _next_batch(self) -> Union[CutSet, Tuple[CutSet]]:
        self.allow_iter_to_reset_state()
        if self.merge_batches:
            # Take a batch from each sampler and merge it.
            # Useful when the Dataset class doesn't treat
            # different sources of cuts in any different way.
            #
            # Note: merging batches is tricky because the samplers can be either
            # SimpleCutSampler or CutPairsSampler, and we need to handle them differently.
            cuts: List[Union[CutSet, Tuple[CutSet]]] = []
            for sampler in self.samplers:
                batch = next(sampler)
                cuts.append(batch)
            if not cuts:
                return CutSet()
            if isinstance(batch, CutSet):
                # Each returned batch is a CutSet -- flatten them.
                return CutSet.from_cuts(c for batch in cuts for c in batch)
            else:
                # Each returned batch is a tuple of CutSets -- determine the tuple size N
                # and merge each N-th CutSet together; return a tuple of merged CutSets.
                tuple_len = len(batch)
                cut_sets = []
                for i in range(tuple_len):
                    cut_sets.append(
                        CutSet.from_cuts(c for batch in cuts for c in batch[i])
                    )
                return tuple(cut_sets)
        else:
            # Take a batch from each sampler and return tuple of batches.
            # Useful when the Dataset treats each source differently.
            cuts: List[CutSet] = []
            for sampler in self.samplers:
                cuts.append(next(sampler))
            return tuple(cuts)

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
