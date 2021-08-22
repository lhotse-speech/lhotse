from functools import reduce
from operator import add
from typing import Callable, Optional

from lhotse import CutSet
from lhotse.cut import Cut
from lhotse.dataset.sampling.base import CutSampler


class ZipSampler(CutSampler):
    """
    :class:`.ZipSampler` takes several samplers as input and concatenates their
    sampled batch cuts together into a single list.
    It is helpful for ensuring that each batch consists of some proportion of cuts
    coming from different sources.

    The input samplers do not have to provide the same number of batches -- when
    any of the samplers becomes depleted, the iteration will stop (like with
    Python's ``zip()`` function).

    Example::

        >>> sampler = ZipSampler(
        ...     SingleCutSampler(cuts_corpusA, max_duration=250, shuffle=True),
        ...     SingleCutSampler(cuts_corpusB, max_duration=100, shuffle=True),
        ... )
        >>> for cut in sampler:
        ...     pass  # profit
    """

    def __init__(self, *samplers: CutSampler) -> None:
        super().__init__()
        self.samplers = samplers

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

    def __iter__(self):
        for sampler in self.samplers:
            iter(sampler)
        return self

    def _next_batch(self) -> CutSet:
        cuts = []
        for sampler in self.samplers:
            cuts.extend(next(sampler))
        return CutSet.from_cuts(cuts)

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

        When set, we will remove the ``__len__`` attribute on the sampler, as it is now
        determined dynamically.

        Example:
            >>> cuts = CutSet(...)
            ... sampler = SingleCutSampler(cuts, max_duration=100.0)
            ... # Retain only the cuts that have at least 1s and at most 20s duration.
            ... sampler.filter(lambda cut: 1.0 <= cut.duration <= 20.0)
        """
        for sampler in self.samplers:
            sampler.filter(predicate)

    def __len__(self):
        if self.num_batches is None:
            self.num_batches = min(len(sampler) for sampler in self.samplers)
        return self.num_batches

    def get_report(self) -> str:
        """Returns a string describing the statistics of the sampling process so far."""
        total_diagnostics = reduce(
            add, (sampler.diagnostics for sampler in self.samplers)
        )
        return total_diagnostics.get_report()
