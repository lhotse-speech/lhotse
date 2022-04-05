import random
import warnings
from collections import deque
from typing import Callable, Generator, Iterable, List, Optional, Tuple, Union

from lhotse import CutSet, Seconds
from lhotse.cut import Cut
from lhotse.dataset.sampling.base import CutSampler, SamplingDiagnostics, TimeConstraint
from lhotse.utils import ifnone, streaming_shuffle


class DynamicCutSampler(CutSampler):
    """
    A dynamic (streaming) variant of sampler that doesn't stratify the sampled cuts in any way.
    It is a generalization of :class:`~lhotse.dataset.sampling.SimpleCutSampler` and
    :class:`~lhotse.dataset.sampling.CutPairsSampler` in that it allows to jointly iterate
    an arbitrary number of CutSets.

    When input CutSets are opened in lazy mode, this sampler doesn't require reading
    the whole cut set into memory.

    For scenarios such as ASR, VAD, Speaker ID, or TTS training, this class supports single CutSet
    iteration. Example::

        >>> cuts = CutSet(...)
        >>> sampler = DynamicCutSampler(cuts, max_duration=100)
        >>> for batch in sampler:
        ...     assert isinstance(batch, CutSet)

    For other scenarios that require pairs (or triplets, etc.) of utterances, this class supports
    zipping multiple CutSets together. Such scenarios could be voice conversion, speech translation,
    contrastive self-supervised training, etc. Example::

        >>> source_cuts = CutSet(...)
        >>> target_cuts = CutSet(...)
        >>> sampler = DynamicCutSampler(source_cuts, target_cuts, max_duration=100)
        >>> for batch in sampler:
        ...     assert isinstance(batch, tuple)
        ...     assert len(batch) == 2
        ...     assert isinstance(batch[0], CutSet)
        ...     assert isinstance(batch[1], CutSet)

    .. note:: for cut pairs, triplets, etc. the user is responsible for ensuring that the CutSets
        are all sorted so that when iterated over sequentially, the items are matched.
        We take care of preserving the right ordering internally, e.g., when shuffling.
        By default, we check that the cut IDs are matching, but that can be disabled.

    .. caution:: when using :meth:`DynamicCutSampler.filter` to filter some cuts with more than
        one CutSet to sample from, we sample one cut from every CutSet, and expect that all of the cuts
        satisfy the predicate -- otherwise, they are all discarded from being sampled.
    """

    def __init__(
        self,
        *cuts: CutSet,
        max_duration: Optional[float] = None,
        max_cuts: Optional[int] = None,
        shuffle: bool = False,
        drop_last: bool = False,
        consistent_ids: bool = True,
        shuffle_buffer_size: int = 20000,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        """
        :param cuts: one or more CutSets (when more than one, will yield tuples of CutSets as mini-batches)
        :param max_duration: The maximum total recording duration from ``cuts``.
            Note: with multiple CutSets, ``max_duration`` constraint applies only to the first CutSet.
        :param max_cuts: The maximum total number of ``cuts`` per batch.
            When only ``max_duration`` is specified, this sampler yields static batch sizes.
        :param shuffle: When ``True``, the cuts will be shuffled dynamically with
            a reservoir-sampling-based algorithm.
            Convenient when mini-batch loop is inside an outer epoch-level loop, e.g.:
            `for epoch in range(10): for batch in dataset: ...` as every epoch will see a
            different cuts order.
        :param drop_last: When ``True``, we will drop all incomplete batches.
            A batch is considered incomplete if it depleted a bucket before
            hitting the constraint such as max_duration, max_cuts, etc.
        :param consistent_ids: Only affects processing of multiple CutSets.
            When ``True``, at each sampling step we check cuts from all CutSets have the same ID
            (i.e., the first cut from every CutSet should have the same ID, same for the second, third, etc.).
        :param shuffle_buffer_size: How many cuts (or cut pairs, triplets) are being held in memory
            a buffer used for streaming shuffling. Larger number means better randomness at the cost
            of higher memory usage.
        :param world_size: Total number of distributed nodes. We will try to infer it by default.
        :param rank: Index of distributed node. We will try to infer it by default.
        :param seed: Random seed used to consistently shuffle the dataset across different processes.
        """
        super().__init__(world_size=world_size, rank=rank, seed=seed)
        if not all(cs.is_lazy for cs in cuts if isinstance(cs, CutSet)):
            warnings.warn(
                "You are using DynamicCutSampler with an eagerly read CutSet. "
                "You won't see any memory/speed benefits with that setup. "
                "Use e.g. 'CutSet.from_jsonl_lazy' to read the CutSet lazily."
            )
        self.cuts = cuts
        self.max_duration = max_duration
        self.max_cuts = max_cuts
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.consistent_ids = consistent_ids
        self.shuffle_buffer_size = shuffle_buffer_size
        self.rng = None

    def __iter__(self) -> "DynamicCutSampler":
        self.rng = random.Random(self.seed + self.epoch)
        # Initiate iteration
        self.cuts_iter = [iter(cs) for cs in self.cuts]
        # Optionally shuffle
        if self.shuffle:
            self.cuts_iter = [
                # Important -- every shuffler has a copy of RNG seeded in the same way,
                # so that they are reproducible.
                streaming_shuffle(
                    cs,
                    rng=random.Random(self.seed + self.epoch),
                    bufsize=self.shuffle_buffer_size,
                )
                for cs in self.cuts_iter
            ]
        # Apply filter predicate
        self.cuts_iter = Filter(
            iterator=zip(*self.cuts_iter),
            predicate=lambda tpl: all(self._filter_fn(c) for c in tpl),
            diagnostics=self.diagnostics,
        )
        # Convert Iterable[Cut] -> Iterable[CutSet]
        self.cuts_iter = DurationBatcher(
            self.cuts_iter,
            max_duration=self.max_duration,
            max_cuts=self.max_cuts,
            drop_last=self.drop_last,
            diagnostics=self.diagnostics,
        )
        self.cuts_iter = iter(self.cuts_iter)
        return self

    def _next_batch(self) -> Union[CutSet, Tuple[CutSet]]:
        batch = next(self.cuts_iter)
        if self.consistent_ids and isinstance(batch, tuple):
            for cuts in zip(*batch):
                expected_id = cuts[0].id
                assert all(c.id == expected_id for c in cuts[1:]), (
                    f"The input CutSet are not sorted by cut ID in the same way. "
                    f"We sampled the following mismatched cut IDs: {', '.join(c.id for c in cuts)}. "
                    f"If this is expected, pass the argument 'consistent_ids=False' to DynamicBucketingSampler."
                )
        return batch

    @property
    def remaining_duration(self) -> Optional[float]:
        return None

    @property
    def remaining_cuts(self) -> Optional[int]:
        return None

    @property
    def num_cuts(self) -> Optional[int]:
        return None


# Note: this class is a subset of SimpleCutSampler and is "datapipes" ready.
class DurationBatcher:
    def __init__(
        self,
        datapipe: Iterable[Union[Cut, Tuple[Cut]]],
        max_frames: int = None,
        max_samples: int = None,
        max_duration: Seconds = None,
        max_cuts: Optional[int] = None,
        drop_last: bool = False,
        strict: bool = False,
        diagnostics: Optional[SamplingDiagnostics] = None,
    ) -> None:
        self.datapipe = datapipe
        self.reuse_cuts_buffer = deque()
        self.drop_last = drop_last
        self.max_cuts = max_cuts
        self.diagnostics = ifnone(diagnostics, SamplingDiagnostics())
        self.time_constraint = TimeConstraint(
            max_duration=max_duration,
            max_frames=max_frames,
            max_samples=max_samples,
            strict=strict,
        )

    def __iter__(self) -> Generator[Union[CutSet, Tuple[CutSet]], None, None]:
        self.cuts_iter = iter(self.datapipe)
        try:
            while True:
                yield self._collect_batch()
        except StopIteration:
            pass
        self.cuts_iter = None

    def _collect_batch(self) -> Union[CutSet, Tuple[CutSet]]:
        def detuplify(
            cuts: List[Union[Cut, Tuple[Cut]]]
        ) -> Union[CutSet, Tuple[CutSet]]:
            """Helper to do the right thing whether we sampled single cuts or cut tuples."""
            if isinstance(cuts[0], tuple):
                if len(cuts[0]) == 1:
                    cuts = CutSet.from_cuts(cs[0] for cs in cuts)
                    self.diagnostics.keep(cuts)
                    return cuts
                else:
                    tuple_of_cut_lists = list(zip(*cuts))
                    self.diagnostics.keep(cuts[0])
                    return tuple([CutSet.from_cuts(cs) for cs in tuple_of_cut_lists])
            else:
                self.diagnostics.keep(cuts)
                return CutSet.from_cuts(cuts)

        self.time_constraint.reset()
        cuts = []
        while True:
            # Check that we have not reached the end of the dataset.
            try:
                if self.reuse_cuts_buffer:
                    next_cut_or_tpl = self.reuse_cuts_buffer.popleft()
                else:
                    # If this doesn't raise (typical case), it's not the end: keep processing.
                    next_cut_or_tpl = next(self.cuts_iter)
            except StopIteration:
                # No more cuts to sample from: if we have a partial batch,
                # we may output it, unless the user requested to drop it.
                # We also check if the batch is "almost there" to override drop_last.
                if cuts and (
                    not self.drop_last or self.time_constraint.close_to_exceeding()
                ):
                    # We have a partial batch and we can return it.
                    return detuplify(cuts)
                else:
                    # There is nothing more to return or it's discarded:
                    # signal the iteration code to stop.
                    try:
                        self.diagnostics.discard(cuts)
                    except AttributeError:  # accounts for cuts being a tuple
                        self.diagnostics.discard(cuts[0])
                    raise StopIteration()

            # Track the duration/frames/etc. constraints.
            self.time_constraint.add(
                next_cut_or_tpl[0]
                if isinstance(next_cut_or_tpl, tuple)
                else next_cut_or_tpl
            )
            next_num_cuts = len(cuts) + 1

            # Did we exceed the max_frames and max_cuts constraints?
            if not self.time_constraint.exceeded() and (
                self.max_cuts is None or next_num_cuts <= self.max_cuts
            ):
                # No - add the next cut to the batch, and keep trying.
                cuts.append(next_cut_or_tpl)
            else:
                # Yes. Do we have at least one cut in the batch?
                if cuts:
                    # Yes. Return the batch, but keep the currently drawn cut for later.
                    self.reuse_cuts_buffer.append(next_cut_or_tpl)
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
                    cuts.append(next_cut_or_tpl)

        return detuplify(cuts)


class Filter(Iterable):
    """
    A wrapper over an iterable that enables lazy filtering.
    It works like Python's `filter` built-in by applying the filter predicate
    to each element and yielding it further if predicate returned ``True``.

    This variant additionally tracks the number of discarded items and updates
    the sampling statistics.
    """

    def __init__(
        self,
        iterator: Iterable,
        predicate: Callable[[Cut], bool],
        diagnostics: Optional[SamplingDiagnostics] = None,
    ) -> None:
        self.iterator = iterator
        self.predicate = predicate
        self.diagnostics = ifnone(diagnostics, SamplingDiagnostics())

        assert callable(
            self.predicate
        ), f"LazyFilter: 'predicate' arg must be callable (got {predicate})."

    def __iter__(self) -> Iterable:
        for item in self.iterator:
            if self.predicate(item):
                yield item
            else:
                self.diagnostics.discard(item)
