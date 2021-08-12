import warnings
from typing import Optional, Tuple

from lhotse import CutSet, Seconds
from lhotse.dataset.sampling.base import CutSampler, TimeConstraint
from lhotse.dataset.sampling.data_source import DataSource


class CutPairsSampler(CutSampler):
    """
    Samples pairs of cuts from a "source" and "target" CutSet.
    It expects that both CutSet's strictly consist of Cuts with corresponding IDs.
    It behaves like an iterable that yields lists of strings (cut IDs).

    When one of :attr:`max_frames`, :attr:`max_samples`, or :attr:`max_duration` is specified,
    the batch size is dynamic.
    Exactly zero or one of those constraints can be specified.
    Padding required to collate the batch does not contribute to max frames/samples/duration.
    """

    def __init__(
            self,
            source_cuts: CutSet,
            target_cuts: CutSet,
            max_source_frames: int = None,
            max_source_samples: int = None,
            max_source_duration: Seconds = None,
            max_target_frames: int = None,
            max_target_samples: int = None,
            max_target_duration: int = None,
            max_cuts: Optional[int] = None,
            shuffle: bool = False,
            drop_last: bool = False,
            world_size: Optional[int] = None,
            rank: Optional[int] = None,
            seed: int = 0,
    ):
        """
        CutPairsSampler's constructor.

        :param source_cuts: the first ``CutSet`` to sample data from.
        :param target_cuts: the second ``CutSet`` to sample data from.
        :param max_source_frames: The maximum total number of feature frames from ``source_cuts``.
        :param max_source_samples: The maximum total number of audio samples from ``source_cuts``.
        :param max_source_duration: The maximum total recording duration from ``source_cuts``.
        :param max_target_frames: The maximum total number of feature frames from ``target_cuts``.
        :param max_target_samples: The maximum total number of audio samples from ``target_cuts``.
        :param max_target_duration: The maximum total recording duration from ``target_cuts``.
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
            provide_len=not source_cuts.is_lazy and not target_cuts.is_lazy,
            shuffle=shuffle,
            world_size=world_size,
            rank=rank,
            seed=seed,
        )
        self.source_cuts = DataSource(source_cuts)
        self.target_cuts = DataSource(target_cuts)
        # Constraints
        self.source_constraints = TimeConstraint(
            max_duration=max_source_duration,
            max_samples=max_source_samples,
            max_frames=max_source_frames,
        )
        self.target_constraints = TimeConstraint(
            max_duration=max_target_duration,
            max_samples=max_target_samples,
            max_frames=max_target_frames,
        )
        self.max_cuts = max_cuts
        self.drop_last = drop_last

    def __iter__(self) -> "CutPairsSampler":
        """
        Prepare the dataset for iterating over a new epoch. Will shuffle the data if requested.
        """
        if self.shuffle:
            self.source_cuts.shuffle(self.seed + self.epoch)
            self.target_cuts.shuffle(self.seed + self.epoch)
        iter(self.source_cuts)
        iter(self.target_cuts)
        self.diagnostics.reset()
        return self

    def _next_batch(self) -> Tuple[CutSet, CutSet]:
        # Keep iterating the underlying CutSets as long as we hit or exceed the constraints
        # provided by user (the max number of source_feats or max number of cuts).
        # Note: no actual data is loaded into memory yet because the manifests contain all the metadata
        # required to do this operation.
        self.source_constraints.reset()
        self.target_constraints.reset()
        source_cuts = []
        target_cuts = []
        while True:
            # Check that we have not reached the end of the dataset.
            try:
                # We didn't - grab the next cut
                next_source_cut = next(self.source_cuts)
                next_target_cut = next(self.target_cuts)
                assert next_source_cut.id == next_target_cut.id, (
                    "Sampled source and target cuts with differing IDs. "
                    "Ensure that your source and target cuts have the same length, "
                    "the same IDs, and the same order."
                )
            except StopIteration:
                # No more cuts to sample from: if we have a partial batch,
                # we may output it, unless the user requested to drop it.
                # We also check if the batch is "almost there" to override drop_last.
                if source_cuts and (
                        not self.drop_last
                        or self.source_constraints.close_to_exceeding()
                        or self.target_constraints.close_to_exceeding()
                ):
                    # We have a partial batch and we can return it.
                    assert len(source_cuts) == len(
                        target_cuts
                    ), "Unexpected state: some cuts in source / target are missing their counterparts..."
                    self.diagnostics.keep(source_cuts)
                    return CutSet.from_cuts(source_cuts), CutSet.from_cuts(target_cuts)
                else:
                    # There is nothing more to return or it's discarded:
                    # signal the iteration code to stop.
                    self.diagnostics.discard(source_cuts)
                    raise StopIteration()

            # Check whether the cuts we're about to sample satisfy optional user-requested predicate.
            if self._filter_fn is not None and (
                    not self._filter_fn(next_source_cut)
                    or not self._filter_fn(next_target_cut)
            ):
                # No - try another one.
                continue

            self.source_constraints.add(next_source_cut)
            self.target_constraints.add(next_target_cut)
            next_num_cuts = len(source_cuts) + 1

            # Did we exceed the max_source_frames and max_cuts constraints?
            if (
                    not self.source_constraints.exceeded()
                    and not self.target_constraints.exceeded()
                    and (self.max_cuts is None or next_num_cuts <= self.max_cuts)
            ):
                # No - add the next cut to the batch, and keep trying.
                source_cuts.append(next_source_cut)
                target_cuts.append(next_target_cut)
            else:
                # Yes. Do we have at least one cut in the batch?
                if source_cuts:
                    # Yes. Return it.
                    self.source_cuts.take_back(next_source_cut)
                    self.target_cuts.take_back(next_target_cut)
                    break
                else:
                    # No. We'll warn the user that the constrains might be too tight,
                    # and return the cut anyway.
                    warnings.warn(
                        "The first cut drawn in batch collection violates one of the max_... constraints"
                        "we'll return it anyway. Consider increasing max_source_frames/max_cuts/etc."
                    )
                    source_cuts.append(next_source_cut)
                    target_cuts.append(next_target_cut)

        assert len(source_cuts) == len(
            target_cuts
        ), "Unexpected state: some cuts in source / target are missing their counterparts..."
        self.diagnostics.keep(source_cuts)
        return CutSet.from_cuts(source_cuts), CutSet.from_cuts(target_cuts)
