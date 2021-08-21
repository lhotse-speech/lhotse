import warnings
from typing import Optional

from lhotse import CutSet, Seconds
from lhotse.dataset.sampling.base import CutSampler, TimeConstraint
from lhotse.dataset.sampling.data_source import DataSource
from lhotse.utils import is_none_or_gt


class SingleCutSampler(CutSampler):
    """
    Samples cuts from a CutSet to satisfy the input constraints.
    It behaves like an iterable that yields lists of strings (cut IDs).

    When one of :attr:`max_frames`, :attr:`max_samples`, or :attr:`max_duration` is specified,
    the batch size is dynamic.
    Exactly zero or one of those constraints can be specified.
    Padding required to collate the batch does not contribute to max frames/samples/duration.

    Example usage::

        >>> dataset = K2SpeechRecognitionDataset(cuts)
        >>> sampler = SingleCutSampler(cuts, shuffle=True)
        >>> loader = DataLoader(dataset, sampler=sampler, batch_size=None)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     sampler.set_epoch(epoch)
        ...     train(loader)

    """

    def __init__(
            self,
            cuts: CutSet,
            max_frames: int = None,
            max_samples: int = None,
            max_duration: Seconds = None,
            max_cuts: Optional[int] = None,
            shuffle: bool = False,
            drop_last: bool = False,
            world_size: Optional[int] = None,
            rank: Optional[int] = None,
            seed: int = 0,
    ):
        """
        SingleCutSampler's constructor.

        :param cuts: the ``CutSet`` to sample data from.
        :param max_frames: The maximum total number of feature frames from ``cuts``.
        :param max_samples: The maximum total number of audio samples from ``cuts``.
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
            provide_len=not cuts.is_lazy,
            shuffle=shuffle,
            world_size=world_size,
            rank=rank,
            seed=seed,
        )
        self.data_source = DataSource(cuts)
        self.time_constraint = TimeConstraint(
            max_duration=max_duration, max_frames=max_frames, max_samples=max_samples
        )
        self.drop_last = drop_last
        self.max_cuts = max_cuts
        assert self.time_constraint.is_active() or not (
                self.time_constraint.is_active() and self.max_cuts is not None
        )
        # Constraints
        assert is_none_or_gt(self.max_cuts, 0)

    @property
    def remaining_duration(self) -> Optional[float]:
        """
        Remaining duration of data left in the sampler (may be inexact due to float arithmetic).
        Not available when the CutSet is read in lazy mode (returns None).
        """
        return self.data_source.remaining_duration

    @property
    def remaining_cuts(self) -> Optional[int]:
        """
        Remaining number of cuts in the sampler.
        Not available when the CutSet is read in lazy mode (returns None).
        """
        return self.data_source.remaining_cuts

    @property
    def num_cuts(self) -> Optional[int]:
        """
        Total number of cuts in the sampler.
        Not available when the CutSet is read in lazy mode (returns None).
        """
        if self.data_source.is_lazy:
            return None
        return len(self.data_source)

    def __iter__(self) -> "SingleCutSampler":
        """
        Prepare the dataset for iterating over a new epoch. Will shuffle the data if requested.
        """
        if self.shuffle:
            self.data_source.shuffle(self.seed + self.epoch)
        iter(self.data_source)
        self.diagnostics.reset()
        return self

    def _next_batch(self) -> CutSet:
        # Keep iterating the underlying CutSet as long as we hit or exceed the constraints
        # provided by user (the max number of frames or max number of cuts).
        # Note: no actual data is loaded into memory yet because the manifests contain all the metadata
        # required to do this operation.
        self.time_constraint.reset()
        cuts = []
        while True:

            # Check that we have not reached the end of the dataset.
            try:
                # If this doesn't raise (typical case), it's not the end: keep processing.
                next_cut = next(self.data_source)
            except StopIteration:
                # No more cuts to sample from: if we have a partial batch,
                # we may output it, unless the user requested to drop it.
                # We also check if the batch is "almost there" to override drop_last.
                if cuts and (
                        not self.drop_last or self.time_constraint.close_to_exceeding()
                ):
                    # We have a partial batch and we can return it.
                    self.diagnostics.keep(cuts)
                    return CutSet.from_cuts(cuts)
                else:
                    # There is nothing more to return or it's discarded:
                    # signal the iteration code to stop.
                    self.diagnostics.discard(cuts)
                    raise StopIteration()

            # Check whether the cut we're about to sample satisfies optional user-requested predicate.
            if self._filter_fn is not None and not self._filter_fn(next_cut):
                # No - try another one.
                continue

            # Track the duration/frames/etc. constraints.
            self.time_constraint.add(next_cut)
            next_num_cuts = len(cuts) + 1

            # Did we exceed the max_frames and max_cuts constraints?
            if not self.time_constraint.exceeded() and (
                    self.max_cuts is None or next_num_cuts <= self.max_cuts
            ):
                # No - add the next cut to the batch, and keep trying.
                cuts.append(next_cut)
            else:
                # Yes. Do we have at least one cut in the batch?
                if cuts:
                    # Yes. Return the batch, but keep the currently drawn cut for later.
                    self.data_source.take_back(next_cut)
                    break
                else:
                    # No. We'll warn the user that the constrains might be too tight,
                    # and return the cut anyway.
                    warnings.warn(
                        "The first cut drawn in batch collection violates "
                        "the max_frames, max_cuts, or max_duration constraints - "
                        "we'll return it anyway. "
                        "Consider increasing max_frames/max_cuts/max_duration."
                    )
                    cuts.append(next_cut)

        self.diagnostics.keep(cuts)
        return CutSet.from_cuts(cuts)
