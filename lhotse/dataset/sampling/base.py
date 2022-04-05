import warnings
from copy import deepcopy
from dataclasses import asdict, dataclass
from math import isclose
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

from torch import distributed as dist
from torch.utils.data import Sampler

from lhotse import Seconds
from lhotse.cut import Cut
from lhotse.utils import exactly_one_not_null, is_none_or_gt


class CutSampler(Sampler):
    """
    ``CutSampler`` is responsible for collecting batches of cuts, given specified criteria.
    It implements correct handling of distributed sampling in ``DataLoader``,
    so that the cuts are not duplicated across workers.

    Sampling in a ``CutSampler`` is intended to be very quick - it only uses the metadata in
    ``CutSet`` manifest to select the cuts, and is not intended to perform any I/O.

    CutSampler works similarly to PyTorch's DistributedSampler - when :attr:`shuffle=True`,
    you should call ``sampler.set_epoch(epoch)`` at each new epoch to have a different
    ordering of returned elements. However, its actual behaviour is different than that of
    DistributedSampler -- instead of partitioning the underlying cuts into equally sized chunks,
    it will return every N-th batch and skip the other batches (where ``N == world_size``).
    The formula used to determine which batches are returned is:
    ``(batch_idx + (world_size - rank)) % world_size == 0``.
    This ensures that we can return an equal number of batches in all distributed workers
    in spite of using a dynamic batch size, at the cost of skipping at most ``world_size - 1`` batches.

    Example usage::

        >>> dataset = K2SpeechRecognitionDataset(cuts)
        >>> sampler = SimpleCutSampler(cuts, max_duration=200, shuffle=True)
        >>> loader = DataLoader(dataset, sampler=sampler, batch_size=None)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     sampler.set_epoch(epoch)
        ...     train(loader)

    .. note::

        For implementers of new samplers:
        Subclasses of CutSampler are expected to implement ``self._next_batch()`` to introduce specific
        sampling logic (e.g. based on filters such as max number of frames/tokens/etc.).
        CutSampler defines ``__iter__()``, which optionally shuffles the cut IDs, and resets
        ``self.cut_idx`` to zero (to be used and incremented inside of ``_next_batch()``.
    """

    def __init__(
        self,
        shuffle: bool = False,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        """
        :param shuffle: When ``True``, the cuts will be shuffled at the start of iteration.
            Convenient when mini-batch loop is inside an outer epoch-level loop, e.g.:
            `for epoch in range(10): for batch in dataset: ...` as every epoch will see a
            different cuts order.
        :param world_size: Total number of distributed nodes. We will try to infer it by default.
        :param rank: Index of distributed node. We will try to infer it by default.
        :param seed: Random seed used to consistently shuffle the dataset across different processes.
        """
        super().__init__(
            data_source=None
        )  # the "data_source" arg is not used in Sampler...
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self._diagnostics = SamplingDiagnostics()

        # This flag is used to indicate that we have restored a sampler's state from a state_dict.
        # When it is set, we will ignore the next call to iter(), which would have reset the
        # iteration state. This way we can resume training exactly from where it was left off.
        self._just_restored_state = False

        self._maybe_init_distributed(world_size=world_size, rank=rank)
        # By default, self._filter_fn passes every Cut through.
        self._filter_fn: Callable[[Cut], bool] = _filter_nothing

    @property
    def diagnostics(self):
        """
        Info on how many cuts / batches were returned or rejected during iteration.

        This property can be overriden by child classes e.g. to merge diagnostics of composite samplers.
        """
        return self._diagnostics

    def _maybe_init_distributed(self, world_size: Optional[int], rank: Optional[int]):
        if world_size is not None:
            assert world_size >= 1
        if rank is not None:
            assert rank >= 0
        if not dist.is_available() or not dist.is_initialized():
            self.world_size = 1 if world_size is None else world_size
            self.rank = 0 if rank is None else rank
            return
        self.world_size = dist.get_world_size() if world_size is None else world_size
        self.rank = dist.get_rank() if rank is None else rank
        assert self.rank < self.world_size

    def set_epoch(self, epoch: int) -> None:
        """
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        :param epoch: Epoch number.
        """
        if self.epoch != epoch:
            # Changing the epoch automatically tells the sampler to discard the progress
            # from a previously read state dict.
            self.allow_iter_to_reset_state()
        self.epoch = epoch
        self.diagnostics.set_epoch(epoch)

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
        self._filter_fn = predicate

    def state_dict(self) -> Dict[str, Any]:
        """
        Return the current state of the sampler in a state_dict.
        Together with ``load_state_dict()``, this can be used to restore the
        training loop's state to the one stored in the state_dict.
        """
        return {
            "epoch": self.epoch,
            "world_size": self.world_size,
            "rank": self.rank,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "diagnostics": self.diagnostics.state_dict(),
        }

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
        world_size = state_dict.pop("world_size")
        assert self.world_size == world_size, (
            f"Cannot restore sampler with a different world_size (before load_state_dict(): {self.world_size},"
            f"attempted restoring to {world_size}). Changing the world_size would result in different batches "
            f"being returned from the sampler."
        )
        # We are explicitly discarding the "rank" argument to support restoring multi-GPU training
        # without too much hassle.
        # We assume that if the world_size is OK, the samplers local ranks are fine.
        del state_dict["rank"]
        assert self.seed == state_dict.pop("seed")
        shuffle = state_dict.pop("shuffle")
        if self.shuffle != shuffle:
            warnings.warn(
                "Overriding the shuffle value in CutSampler based on state_dict"
                f"(initialized to {self.shuffle}; restored to {shuffle})."
            )
        self.shuffle = shuffle
        self.epoch = state_dict.pop("epoch")
        self.diagnostics.load_state_dict(state_dict.pop("diagnostics"))
        assert (
            len(state_dict) == 0
        ), "Error in CutSampler.load_state_dict(): Unexpected keys:\n- " + "\n- ".join(
            state_dict.keys()
        )
        self._just_restored_state = True

    def __iter__(self):
        raise NotImplementedError(
            "Sub-classes of CutSampler have to implement __iter__()"
        )

    def _next_batch(self):
        raise NotImplementedError(
            "Sub-classes of CutSampler have to implement self._next_batch()"
        )

    @property
    def remaining_duration(self) -> Optional[float]:
        """
        Remaining duration of data left in the sampler (may be inexact due to float arithmetic).
        Not available when the CutSet is read in lazy mode (returns None).
        """
        raise NotImplementedError(
            "Sub-classes of CutSampler have to implement self.remaining_duration"
        )

    @property
    def remaining_cuts(self) -> Optional[int]:
        """
        Remaining number of cuts in the sampler.
        Not available when the CutSet is read in lazy mode (returns None).
        """
        raise NotImplementedError(
            "Sub-classes of CutSampler have to implement self.remaining_cuts"
        )

    @property
    def num_cuts(self) -> Optional[int]:
        """
        Total number of cuts in the sampler.
        Not available when the CutSet is read in lazy mode (returns None).
        """
        raise NotImplementedError(
            "Sub-classes of CutSampler have to implement self.num_cuts"
        )

    def allow_iter_to_reset_state(self):
        """
        Enables re-setting to the start of an epoch when iter() is called.
        This is only needed in one specific scenario: when we restored previous
        sampler state via ``sampler.load_state_dict()`` but want to discard
        the progress in the current epoch and start from the beginning.
        """
        self._just_restored_state = False

    def __next__(self):
        self.allow_iter_to_reset_state()
        # We use the following trick to ensure equal number of batches for each distributed
        # worker:
        # Every time a next batch is required, we will sample self.world_size batches first,
        # and then return the one at position self.rank.
        # This way, if any of the batches raises StopIteration, we'll know to stop early
        # when a given batch was available for one of the nodes, but not for the others.
        batches = []
        for _ in range(self.world_size):
            batches.append(self._next_batch())
        return batches[self.rank]

    def get_report(self) -> str:
        """Returns a string describing the statistics of the sampling process so far."""
        return self.diagnostics.get_report()


@dataclass
class TimeConstraint:
    """
    Represents a time-based constraint for sampler classes.
    It can be defined either as maximum total batch duration (in seconds),
    number of frames, or number of samples.
    These options are mutually exclusive and this class checks for that.

    :class:`TimeConstraint` can be used for tracking whether the criterion has been exceeded
    via the `add(cut)`, `exceeded()` and `reset()` methods.
    It will automatically track the right criterion (i.e. select frames/samples/duration from the cut).
    It can also be a null constraint (never exceeded).
    """

    max_duration: Optional[Seconds] = None
    max_samples: Optional[int] = None
    max_frames: Optional[int] = None
    current: Union[int, Seconds] = 0
    num_cuts: int = 0
    longest_seen: Union[int, float] = 0
    strict: bool = False

    def __post_init__(self) -> None:
        assert exactly_one_not_null(*self._constraints) or all(
            x is None for x in self._constraints
        )
        for c in self._constraints:
            assert is_none_or_gt(c, 0)

    @property
    def _constraints(self) -> Tuple:
        return self.max_duration, self.max_frames, self.max_samples

    @property
    def active_constraint(self) -> Union[int, float, None]:
        if self.max_frames is not None:
            return self.max_frames
        if self.max_samples is not None:
            return self.max_samples
        if self.max_duration is not None:
            return self.max_duration
        return None

    def is_active(self) -> bool:
        """Is it an actual constraint, or a dummy one (i.e. never exceeded)."""
        return any(x is not None for x in self._constraints)

    def add(self, cut: Cut) -> None:
        """
        Increment the internal counter for the time constraint,
        selecting the right property from the input ``cut`` object.
        """
        if self.max_frames is not None:
            self.current += cut.num_frames
            self.longest_seen = max(self.longest_seen, cut.num_frames)
        if self.max_samples is not None:
            self.current += cut.num_samples
            self.longest_seen = max(self.longest_seen, cut.num_samples)
        if self.max_duration is not None:
            self.current += cut.duration
            self.longest_seen = max(self.longest_seen, cut.duration)
        self.num_cuts += 1

    def exceeded(self) -> bool:
        """Is the constraint exceeded or not."""
        constraint = self.active_constraint
        if constraint is None:
            return False
        if self.strict:
            return self.num_cuts * self.longest_seen > constraint
        else:
            return self.current > constraint

    def close_to_exceeding(self) -> bool:
        """
        Check if the batch is close to satisfying the constraints.
        We define "closeness" as: if we added one more cut that has
        duration/num_frames/num_samples equal to the mean of the current
        batch, then the batch would have exceeded the constraints.
        """
        if self.strict:
            thresh = self.longest_seen
        else:
            thresh = self.current / self.num_cuts

        if self.max_frames is not None:
            return self.current + thresh > self.max_frames
        if self.max_samples is not None:
            return self.current + thresh > self.max_samples
        if self.max_duration is not None:
            return self.current + thresh > self.max_duration
        return False

    def reset(self) -> None:
        """
        Reset the internal counter (to be used after a batch was created,
        to start collecting a new one).
        """
        self.current = 0
        self.num_cuts = 0
        self.longest_seen = 0

    def state_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.max_duration = state_dict.pop("max_duration")
        self.max_samples = state_dict.pop("max_samples")
        self.max_frames = state_dict.pop("max_frames")
        self.current = state_dict.pop("current")
        self.num_cuts = state_dict.pop("num_cuts")
        self.strict = state_dict.pop("strict", False)
        self.longest_seen = state_dict.pop("longest_seen", 0)
        assert len(state_dict) == 0, (
            "Error in TimeConstraint.load_state_dict(): Unexpected keys:\n- "
            + "\n- ".join(state_dict.keys())
        )

    def __add__(self, other: "TimeConstraint") -> "TimeConstraint":
        for key in ("max_duration", "max_frames", "max_samples"):
            self_attr = getattr(self, key)
            other_attr = getattr(other, key)
            is_none = self_attr is None and other_attr is None
            assert is_none or isclose(self_attr, other_attr), (
                f"To add two TimeConstraint objects, they need to represent the same constraint "
                f"(got self.{key}={self_attr} != other.{key}={other_attr})."
            )
        assert self.strict == other.strict
        return TimeConstraint(
            max_duration=self.max_duration,
            max_frames=self.max_frames,
            max_samples=self.max_samples,
            current=self.current + other.current,
            num_cuts=self.num_cuts + other.num_cuts,
            strict=self.strict,
            longest_seen=max(self.longest_seen, other.longest_seen),
        )

    def __eq__(self, other: "TimeConstraint") -> bool:
        return (
            self.max_duration == other.max_duration
            and self.max_samples == other.max_samples
            and self.max_frames == other.max_frames
        )


@dataclass
class EpochDiagnostics:
    epoch: int = 0
    kept_cuts: int = 0
    discarded_cuts: int = 0
    kept_batches: int = 0
    discarded_batches: int = 0

    @property
    def total_cuts(self) -> int:
        return self.kept_cuts + self.discarded_cuts

    @property
    def total_batches(self) -> int:
        return self.kept_batches + self.discarded_batches

    def get_report(self) -> str:
        """Returns a string describing the statistics of the sampling process so far."""
        if self.total_batches == 0 or self.total_cuts == 0:
            return (
                "Sampling statistics unavailable: EpochDiagnostics received no cuts or batches. "
                "If this is unexpected, and you're using a custom sampler, ensure that the sampler "
                "is registering the batches in SamplerDiagnostics/EpochDiagnostics."
            )

        return (
            f"| ep {self.epoch:>3d} | cuts kept {self.kept_cuts:d}/{self.total_cuts:d} "
            f"({self.kept_cuts / self.total_cuts:.2%}) "
            f"| cuts discarded {self.discarded_cuts:d} "
            f"| batches kept {self.kept_batches:d}/{self.total_batches:d} "
            f"({self.kept_batches / self.total_batches:.2%})"
            f"| batches discarded {self.discarded_batches:d} |"
        )

    def state_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> "EpochDiagnostics":
        self.epoch = state_dict.pop("epoch")
        self.kept_batches = state_dict.pop("kept_batches")
        self.discarded_batches = state_dict.pop("discarded_batches")
        self.kept_cuts = state_dict.pop("kept_cuts")
        self.discarded_cuts = state_dict.pop("discarded_cuts")

        assert len(state_dict) == 0, (
            "Error in EpochDiagnostics.load_state_dict(): Unexpected keys:\n- "
            + "\n- ".join(state_dict.keys())
        )

        return self

    def __add__(self, other: "EpochDiagnostics") -> "EpochDiagnostics":
        assert self.epoch == other.epoch
        return EpochDiagnostics(
            epoch=self.epoch,
            kept_cuts=self.kept_cuts + other.kept_cuts,
            kept_batches=self.kept_batches + other.kept_batches,
            discarded_cuts=self.discarded_cuts + other.discarded_cuts,
            discarded_batches=self.discarded_batches + other.discarded_batches,
        )


@dataclass
class SamplingDiagnostics:
    """
    Utility for collecting diagnostics about the sampling process:
    how many cuts/batches were discarded.
    """

    current_epoch: int = 0
    stats_per_epoch: Dict[int, EpochDiagnostics] = None

    def __post_init__(self):
        if self.stats_per_epoch is None:
            self.stats_per_epoch = {}
            self.set_epoch(self.current_epoch)

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch
        if epoch not in self.stats_per_epoch:
            self.stats_per_epoch[epoch] = EpochDiagnostics(epoch=epoch)

    def advance_epoch(self) -> None:
        self.set_epoch(self.current_epoch + 1)

    @property
    def current_epoch_stats(self) -> EpochDiagnostics:
        return self.stats_per_epoch[self.current_epoch]

    def keep(self, cuts: Iterable[Cut]) -> None:
        cntr = 0
        for cut in cuts:
            self.current_epoch_stats.kept_cuts += 1
            cntr += 1
        if not cntr:
            warnings.warn(
                "Found and accepted batch with zero cuts. This could be an error."
            )
        self.current_epoch_stats.kept_batches += 1

    def discard(self, cuts: Iterable[Cut]) -> None:
        cntr = 0
        for cut in cuts:
            self.current_epoch_stats.discarded_cuts += 1
            cntr += 1
        if cntr:
            # We don't warn about discarded batches with 0 cuts.
            self.current_epoch_stats.discarded_batches += 1

    def discard_single(self, cut: Cut) -> None:
        self.current_epoch_stats.discarded_cuts += 1

    @property
    def kept_cuts(self) -> int:
        return sum(s.kept_cuts for s in self.stats_per_epoch.values())

    @property
    def discarded_cuts(self) -> int:
        return sum(s.discarded_cuts for s in self.stats_per_epoch.values())

    @property
    def kept_batches(self) -> int:
        return sum(s.kept_batches for s in self.stats_per_epoch.values())

    @property
    def discarded_batches(self) -> int:
        return sum(s.discarded_batches for s in self.stats_per_epoch.values())

    @property
    def total_cuts(self) -> int:
        return sum(s.total_cuts for s in self.stats_per_epoch.values())

    @property
    def total_batches(self) -> int:
        return sum(s.total_batches for s in self.stats_per_epoch.values())

    def get_report(self, per_epoch: bool = False) -> str:
        """Returns a string describing the statistics of the sampling process so far."""
        if self.total_batches == 0 or self.total_cuts == 0:
            return (
                "Sampling statistics unavailable: the SamplerDiagnostics received no cuts or batches. "
                "If this is unexpected, and you're using a custom sampler, ensure that the sampler "
                "is registering the batches in SamplerDiagnostics."
            )

        ret = []

        if per_epoch:
            for epoch in sorted(self.stats_per_epoch):
                ret.append(self.stats_per_epoch[epoch].get_report())

        ret.append(
            f"|  total  | cuts kept {self.kept_cuts:d}/{self.total_cuts:d} "
            f"({self.kept_cuts / self.total_cuts:.2%}) "
            f"| cuts discarded {self.discarded_cuts:d} "
            f"| batches kept {self.kept_batches:d}/{self.total_batches:d} "
            f"({self.kept_batches / self.total_batches:.2%})"
            f"| batches discarded {self.discarded_batches:d} |"
        )

        return "\n".join(ret)

    def state_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> "SamplingDiagnostics":
        self.current_epoch = state_dict.pop("current_epoch")
        self.stats_per_epoch = {
            epoch: EpochDiagnostics().load_state_dict(sd)
            for epoch, sd in state_dict.pop("stats_per_epoch").items()
        }
        return self

    def __add__(self, other: "SamplingDiagnostics") -> "SamplingDiagnostics":
        stats_per_epoch = deepcopy(self.stats_per_epoch)
        for epoch, stats in other.stats_per_epoch.items():
            if epoch in stats_per_epoch:
                stats_per_epoch[epoch] = stats_per_epoch[epoch] + stats
            else:
                stats_per_epoch[epoch] = stats

        return SamplingDiagnostics(
            current_epoch=self.current_epoch, stats_per_epoch=stats_per_epoch
        )


def _filter_nothing(cut: Cut) -> bool:
    return True
