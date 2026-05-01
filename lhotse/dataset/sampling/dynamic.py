import random
import warnings
from collections import deque
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

from lhotse import CutSet, Seconds
from lhotse.cut import Cut
from lhotse.dataset.dataloading import resolve_seed
from lhotse.dataset.sampling.base import (
    CutSampler,
    SamplingConstraint,
    SamplingDiagnostics,
    TimeConstraint,
    capture_sources_state,
    restore_sources_state,
)
from lhotse.dataset.sampling.checkpoint_backends import (
    build_dynamic_cut_checkpoint_backend,
)
from lhotse.lazy import LazyShuffler, resolve_iterator_source
from lhotse.utils import ifnone


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
        *cuts: Iterable,
        max_duration: Optional[Seconds] = None,
        max_cuts: Optional[int] = None,
        constraint: Optional[SamplingConstraint] = None,
        shuffle: bool = False,
        drop_last: bool = False,
        consistent_ids: bool = True,
        shuffle_buffer_size: int = 20000,
        quadratic_duration: Optional[Seconds] = None,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        seed: Union[int, Literal["trng", "randomized"]] = 0,
        strict=None,
    ) -> None:
        """
        :param cuts: one or more CutSets (when more than one, will yield tuples of CutSets as mini-batches)
        :param max_duration: The maximum total recording duration from ``cuts``.
            Note: with multiple CutSets, ``max_duration`` constraint applies only to the first CutSet.
        :param max_cuts: The maximum total number of ``cuts`` per batch.
            When only ``max_duration`` is specified, this sampler yields static batch sizes.
        :param constraint: Provide a :class:`~lhotse.dataset.sampling.base.SamplingConstraint` object
            defining how the sampler decides when a mini-batch is complete. It also affects which
            attribute of the input examples decides the "size" of the example (by default it's ``.duration``).
            Before this parameter was introduced, Lhotse samplers used
            :class:`~lhotse.dataset.sampling.base.TimeConstraint` implicitly.
            Introduced in Lhotse v1.22.0.
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
        :param quadratic_duration: When set, it adds an extra penalty that's quadratic in size w.r.t.
            a cuts duration. This helps get a more even GPU utilization across different input lengths
            when models have quadratic input complexity. Set between 15 and 40 for transformers.
        :param world_size: Total number of distributed nodes. We will try to infer it by default.
        :param rank: Index of distributed node. We will try to infer it by default.
        :param seed: Random seed used to consistently shuffle the dataset across different processes.
        """
        super().__init__(
            drop_last=drop_last, world_size=world_size, rank=rank, seed=seed
        )
        if not all(cs.is_lazy for cs in cuts if isinstance(cs, CutSet)):
            warnings.warn(
                "You are using DynamicCutSampler with an eagerly read CutSet. "
                "You won't see any memory/speed benefits with that setup. "
                "Use e.g. 'CutSet.from_jsonl_lazy' to read the CutSet lazily."
            )
        self.cuts = cuts
        self.max_duration = max_duration
        self.max_cuts = max_cuts
        self.constraint = constraint
        self.shuffle = shuffle
        self.consistent_ids = consistent_ids
        self.shuffle_buffer_size = shuffle_buffer_size
        self.quadratic_duration = quadratic_duration
        self._active_cuts = None

        if strict is not None:
            warnings.warn(
                "In Lhotse v1.4 all samplers act as if 'strict=True'. "
                "Sampler's argument 'strict' will be removed in a future Lhotse release.",
                category=DeprecationWarning,
            )

    def state_dict(self) -> Dict[str, Any]:
        # The custom-constraint object itself is not serialized: constraints are
        # reconstructed from config on each run. We still capture the iteration
        # state (epoch, diagnostics, source iterator graph) which is what drives
        # exact resume.
        sd = super().state_dict()
        sd.update(
            {
                "max_duration": self.max_duration,
                "max_cuts": self.max_cuts,
                "consistent_ids": self.consistent_ids,
                "shuffle_buffer_size": self.shuffle_buffer_size,
                "quadratic_duration": self.quadratic_duration,
            }
        )
        return sd

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        self.max_duration = sd.pop("max_duration")
        self.max_cuts = sd.pop("max_cuts")
        self.consistent_ids = sd.pop("consistent_ids")
        self.shuffle_buffer_size = sd.pop("shuffle_buffer_size")
        self.quadratic_duration = sd.pop("quadratic_duration")
        sd.pop("strict", None)  # backward compatibility
        super().load_state_dict(sd)
        # Defer _fast_forward to __iter__ so the sampler remains picklable
        # for DataLoader with num_workers > 0.
        self._needs_fast_forward = True

    def _fast_forward(self):
        current_epoch = self.diagnostics.current_epoch
        num_batches_to_iter = self.diagnostics.current_epoch_stats.total_batches

        # Set the right epoch
        self.set_epoch(current_epoch)
        backend = build_dynamic_cut_checkpoint_backend(
            self,
            current_epoch=current_epoch,
            num_batches_to_iter=num_batches_to_iter,
        )
        backend.restore()

    def _initialize_replay_iterator(self) -> None:
        self._cuts_state = None
        self._just_restored_state = False
        self._active_cuts = None
        self._initialize_epoch_iterator(rebuild_sources=True)

    def _replay_step(self) -> None:
        next(self)

    def _make_epoch_sources(self):
        if not self.shuffle:
            return list(self.cuts)

        seed = resolve_seed(self.seed)
        epoch_sources = []
        for src in self.cuts:
            shuffler = LazyShuffler(
                resolve_iterator_source(src),
                buffer_size=self.shuffle_buffer_size,
                rng=random.Random(seed + self.epoch),
            )
            if isinstance(src, CutSet):
                epoch_sources.append(CutSet(shuffler))
            else:
                epoch_sources.append(shuffler)
        return epoch_sources

    def _initialize_epoch_iterator(self, *, rebuild_sources: bool) -> None:
        if rebuild_sources or self._active_cuts is None:
            self._active_cuts = self._make_epoch_sources()
        self.cuts_iter = [iter(resolve_iterator_source(cs)) for cs in self._active_cuts]
        self.cuts_iter = Filter(
            iterator=zip(*self.cuts_iter),
            predicate=lambda tpl: all(self._filter_fn(c) for c in tpl),
            diagnostics=self.diagnostics,
        )
        self.cuts_iter = DurationBatcher(
            self.cuts_iter,
            max_duration=self.max_duration,
            max_cuts=self.max_cuts,
            constraint=self.constraint,
            drop_last=self.drop_last,
            quadratic_duration=self.quadratic_duration,
            diagnostics=self.diagnostics,
        )
        self.cuts_iter = iter(self.cuts_iter)

    def _capture_cuts_state(self) -> Optional[list]:
        sources = self._active_cuts if self._active_cuts is not None else self.cuts
        return capture_sources_state(sources)

    def _restore_cuts_state(self, cuts_state: list) -> None:
        self._active_cuts = self._make_epoch_sources()
        restore_sources_state(self._active_cuts, cuts_state)

    def __iter__(self) -> "DynamicCutSampler":
        if getattr(self, "_needs_fast_forward", False):
            self._needs_fast_forward = False
            self._fast_forward()
            return self
        if self._just_restored_state:
            return self
        # Why reset the current epoch?
        # Either we are iterating the epoch for the first time and it's a no-op,
        # or we are iterating the same epoch again, in which case setting more steps
        # than are actually available per epoch would have broken the checkpoint restoration.
        if getattr(self, "_skip_diagnostics_reset_once", False):
            self._skip_diagnostics_reset_once = False
        else:
            self.diagnostics.reset_current_epoch()
        self._initialize_epoch_iterator(rebuild_sources=True)
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
        max_duration: Seconds = None,
        max_cuts: Optional[int] = None,
        constraint: Optional[SamplingConstraint] = None,
        drop_last: bool = False,
        quadratic_duration: Optional[Seconds] = None,
        diagnostics: Optional[SamplingDiagnostics] = None,
    ) -> None:
        self.datapipe = datapipe
        self.reuse_cuts_buffer = deque()
        self.drop_last = drop_last
        self.diagnostics = ifnone(diagnostics, SamplingDiagnostics())
        check_constraint(constraint, max_duration, max_cuts)
        if constraint is not None:
            self.constraint = constraint
        else:
            self.constraint = TimeConstraint(
                max_duration=max_duration,
                max_cuts=max_cuts,
                quadratic_duration=quadratic_duration,
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
                    return cuts
                else:
                    tuple_of_cut_lists = list(zip(*cuts))
                    return tuple([CutSet.from_cuts(cs) for cs in tuple_of_cut_lists])
            else:
                return CutSet.from_cuts(cuts)

        self.constraint.reset()
        cuts = []
        while True:
            # Check that we have not reached the end of the dataset.
            try:
                # If this doesn't raise (typical case), it's not the end: keep processing.
                next_cut_or_tpl = next(self.cuts_iter)
            except StopIteration:
                # No more cuts to sample from: if we have a partial batch,
                # we may output it, unless the user requested to drop it.
                # We also check if the batch is "almost there" to override drop_last.
                if cuts and (
                    not self.drop_last or self.constraint.close_to_exceeding()
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
            cuts.append(next_cut_or_tpl)
            self.constraint.add(
                next_cut_or_tpl[0]
                if isinstance(next_cut_or_tpl, tuple)
                else next_cut_or_tpl
            )

            # Did we exceed the max_duration and max_cuts constraints?
            if self.constraint.close_to_exceeding():
                # Yes. Finish sampling this batch.
                if self.constraint.exceeded() and len(cuts) == 1:
                    warnings.warn(
                        "We have exceeded the max_duration constraint during sampling but have only 1 cut. "
                        "This is likely because max_duration was set to a very low value ~10s, "
                        "or you're using a CutSet with very long cuts (e.g. 100s of seconds long)."
                    )
                break

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


def check_constraint(constraint: Optional, max_duration: Optional, max_cuts: Optional):
    if constraint is not None:
        assert (
            max_duration is None and max_cuts is None
        ), "Cannot specify both constraint= and max_duration=/max_cuts="
    else:
        assert (
            max_duration is not None or max_cuts is not None
        ), "At least one of max_duration= or max_cuts= has to be defined (or provide constraint=)."
