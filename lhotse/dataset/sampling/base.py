import copy
import os
import warnings
from abc import ABCMeta, abstractmethod
from bisect import bisect_left
from copy import deepcopy
from dataclasses import asdict, dataclass
from math import isclose
from typing import Any, Callable, Dict, Iterable, Literal, Optional, Tuple, Union

import torch
from torch import distributed as dist
from torch.utils.data import Sampler

from lhotse.cut import Cut, CutSet
from lhotse.cut.text import TextExample
from lhotse.lazy import Dillable
from lhotse.manipulation import combine
from lhotse.utils import Seconds, exactly_one_not_null, ifnone, is_none_or_gt


class CutSampler(Sampler, Dillable):
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
        drop_last: bool = False,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        seed: Union[int, Literal["randomized", "trng"]] = 0,
    ) -> None:
        """
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
            data_source=None
        )  # the "data_source" arg is not used in Sampler...
        self.drop_last = drop_last
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
        self._filter_fn: Callable[[Cut], bool] = _filter_nothing()
        self._transforms = []

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

        # Order of precedence:
        # 1. When world size or rank are explicitly provided, we will use them.
        # 2. Next, check WORLD_SIZE and RANK env variables; yes? use them.
        # 3. Next, check if torch.distributed is initialized and has them set; yes? use them.
        # 4. If none of those are available, rank=0 and world_size=1.
        if "WORLD_SIZE" in os.environ and "RANK" in os.environ:
            # If deepspeed launcher is being used, it will set the env variables automatically.
            self.world_size = ifnone(world_size, int(os.environ["WORLD_SIZE"]))
            self.rank = ifnone(rank, int(os.environ["RANK"]))
        elif dist.is_available() and dist.is_initialized():
            self.world_size = ifnone(world_size, dist.get_world_size())
            self.rank = ifnone(rank, dist.get_rank())
        else:
            self.world_size = ifnone(world_size, 1)
            self.rank = ifnone(rank, 0)
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

    def filter(self, predicate: Callable[[Cut], bool]) -> "CutSampler":
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
        if isinstance(self._filter_fn, _filter_nothing):
            self._filter_fn = predicate
        else:
            self._filter_fn = _and(self._filter_fn, predicate)
        return self

    def map(self, fn: Callable[[CutSet], CutSet]) -> "CutSampler":
        """Apply ``fn`` to each mini-batch of ``CutSet`` before yielding it."""
        assert callable(
            fn
        ), f"Expected a callable accepting and returning a CutSet, received: '{fn}'"
        self._transforms.append(fn)
        return self

    def state_dict(self) -> Dict[str, Any]:
        """
        Return the current state of the sampler in a state_dict.
        Together with ``load_state_dict()``, this can be used to restore the
        training loop's state to the one stored in the state_dict.
        """
        return {
            "epoch": self.epoch,
            "drop_last": self.drop_last,
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
        self.drop_last = state_dict.pop("drop_last")
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
        # When world_size=1 (i.e., single device) this doesn't change anything.
        # When world_size>1 (i.e., multi-GPU) the behavior depends on the setting of ``drop_last``.
        # To prevent some rank from terminating the iteration earlier than others (and typically going into deadlock),
        # we will either:
        # a) [drop_last=False, default] redistribute the examples to yield a (partial) mini-batch in each rank
        #       (if there's not enough examples, we'll duplicate some);
        # b) [drop_last=True] we'll stop early and discard the mini-batches for all ranks that had them.
        #       Note that drop_last=True implies the last partial mini-batch would also be discarded in a
        #       single-GPU setting, to be consistent with PyTorch's drop_last logic.
        batches = []
        for _ in range(self.world_size):
            try:
                batch = self._next_batch()
                batches.append(batch)
            except StopIteration:
                if self.world_size == 1 or self.drop_last:
                    # The users indicated they want an equal number of batches on all
                    # ranks and are ready to lose some data: drop remainder batches.
                    raise
                # If we got here, it means there's one or more empty mini-batch that
                # hasn't triggered StopIteration. This scenario is handled below.

        if len(batches) == 0:
            raise StopIteration()  # normal end of iteration when drop_last=False
        elif len(batches) != self.world_size:
            # "From each according to his ability, to each according to his needs."
            # We hit the end of data and at least one rank is left without a mini-batch.
            # Since we have access to the mini-batches in all ranks here, we can
            # deterministically re-distribute the examples to yield a partial mini-batch
            # in every rank (i.e., the result of the redistribution doesn't depend on rank).
            # The only problematic scenario is when the number of examples is smaller
            # than world size. In these cases, we will duplicate the first ``n_diff``
            # examples so that each rank has exactly 1 example in its mini-batch.
            combined = combine([b for b in batches if b is not None])
            chunk = 0
            while (diff := self.world_size - len(combined)) > 0:
                combined = combined + combined.subset(first=diff).modify_ids(
                    mark_as_duplicate(chunk)
                )
                chunk += 1
            batches = combined.split(self.world_size)

        selected = batches[self.rank]
        self._log_diagnostics(selected)
        for tfn in self._transforms:
            selected = tfn(selected)
        attach_dataloading_info(selected, rank=self.rank, world_size=self.world_size)
        return selected

    def _log_diagnostics(self, batch: Union[CutSet, Tuple[CutSet, ...]]) -> None:
        if isinstance(batch, CutSet):
            self.diagnostics.keep(batch)
        elif isinstance(batch, tuple) and isinstance(batch[0], CutSet):
            self.diagnostics.keep(batch[0])
        else:
            raise ValueError(f"Object with unexpected type: {batch}")

    def get_report(self) -> str:
        """Returns a string describing the statistics of the sampling process so far."""
        return self.diagnostics.get_report()


def mark_as_duplicate(iteration: int) -> Callable[[str], str]:
    def inner(cut_id: str) -> str:
        return f"{cut_id}_dup{iteration}"

    return inner


def attach_dataloading_info(cuts: CutSet, rank: int, world_size: int) -> None:
    """
    Attaches diagnostic info about dataloading to each cut under ``dataloading_info`` custom field.
    This information contains the rank, world_size, and worker_id.
    If the training is not distributed, rank and world_size are 0 and 1.
    If the num_workers argument in DataLoader was 0, worker_id is None.
    """
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        worker_id = None
    else:
        worker_id = worker_info.id
    info = {"rank": rank, "world_size": world_size, "worker_id": worker_id}
    for cut in cuts:
        cut.dataloading_info = info


class SamplingConstraint(metaclass=ABCMeta):
    """
    Defines the interface for sampling constraints. A sampling constraint
    keeps track of the sampled examples and lets the sampler know when it
    should yield a mini-batch.
    """

    @abstractmethod
    def add(self, example: Any) -> None:
        """
        Update the sampling constraint with the information about the sampled example
        (e.g. current batch size, total duration).
        """
        pass

    @abstractmethod
    def exceeded(self) -> bool:
        """Inform if the sampling constraint has been exceeded."""
        pass

    @abstractmethod
    def close_to_exceeding(self) -> bool:
        """Inform if we're going to exceed the sampling constraint after adding one more example."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Resets the internal state (called after yielding a mini-batch)."""
        pass

    @abstractmethod
    def measure_length(self, example: Any) -> float:
        """
        Returns the "size" of an example, used to create bucket distribution for bucketing samplers
        (e.g., for audio it may be duration; for text it may be number of tokens; etc.).
        """
        pass

    def select_bucket(
        self, buckets: Any, example: Any = None, example_len: Any = None
    ) -> int:
        """
        Given a list of buckets and an example, assign the example to the correct bucket.
        This is leveraged by bucketing samplers.

        Default implementation assumes that buckets are expressed in the same units as
        the output of :meth:`SamplingConstraint.measure_length` and returns the index
        of the first bucket that has a larger length than the example.
        """
        assert exactly_one_not_null(
            example, example_len
        ), f"select_bucket requires either example= or example_len= as the input (we received {example=} and {example_len=})."
        if example_len is None:
            example_len = self.measure_length(example)
        return bisect_left(buckets, example_len)

    def copy(self) -> "SamplingConstraint":
        """Return a shallow copy of this constraint."""
        return copy.copy(self)


@dataclass
class TimeConstraint(SamplingConstraint):
    """
    Represents a time-based constraint for sampler classes.
    It is defined as maximum total batch duration (in seconds) and/or the total number of cuts.

    :class:`TimeConstraint` can be used for tracking whether the criterion has been exceeded
    via the `add(cut)`, `exceeded()` and `reset()` methods.
    It will automatically track the right criterion (i.e. select duration from the cut).
    It can also be a null constraint (never exceeded).

    When ``quadratic_duration`` is set, we will try to compensate for models that have a
    quadratic complexity w.r.t. the input sequence length. We use the following formula
    to determine the effective duration for each cut::

        effective_duration = duration + (duration ** 2) / quadratic_duration

    We recomend setting quadratic_duration to something between 15 and 40 for transformer architectures.
    """

    max_duration: Optional[Seconds] = None
    max_cuts: Optional[int] = None
    current: Union[int, Seconds] = 0
    num_cuts: int = 0
    longest_seen: Union[int, float] = 0
    quadratic_duration: Optional[Seconds] = None

    def __post_init__(self) -> None:
        assert is_none_or_gt(self.max_duration, 0)
        assert is_none_or_gt(self.max_cuts, 0)
        assert is_none_or_gt(self.quadratic_duration, 0)

    def is_active(self) -> bool:
        """Is it an actual constraint, or a dummy one (i.e. never exceeded)."""
        return self.max_duration is not None or self.max_cuts is not None

    def add(self, example: Cut) -> None:
        """
        Increment the internal counter for the time constraint,
        selecting the right property from the input ``cut`` object.
        """
        if self.max_duration is not None:
            duration = self._maybe_apply_quadratic_correction(example.duration)
            self.current += duration
            self.longest_seen = max(self.longest_seen, duration)
        self.num_cuts += 1

    def _maybe_apply_quadratic_correction(self, duration: Seconds) -> Seconds:
        if self.quadratic_duration is None:
            return duration
        # For the quadratic complexity case, we add a term that accounts for
        # extra memory occupied by the model. The 1/quadratic_duration term causes
        # the effective duration to be doubled when it's equal to quadratic_duration.
        return duration + (duration**2) / self.quadratic_duration

    def exceeded(self) -> bool:
        """Is the constraint exceeded or not."""
        if self.max_cuts is not None and self.num_cuts > self.max_cuts:
            return True
        if self.max_duration is None:
            return False
        effective_duration = self.num_cuts * self.longest_seen
        return effective_duration > self.max_duration

    def close_to_exceeding(self) -> bool:
        """
        Check if the batch is close to satisfying the constraints.
        We define "closeness" as: if we added one more cut that has
        duration/num_frames/num_samples equal to the longest seen cut
        in the current batch, then the batch would have exceeded the constraints.
        """
        if self.max_cuts is not None and self.num_cuts >= self.max_cuts:
            return True

        if self.max_duration is not None:
            effective_duration = (self.num_cuts + 1) * self.longest_seen
            return effective_duration > self.max_duration
        return False

    def reset(self) -> None:
        """
        Reset the internal counter (to be used after a batch was created,
        to start collecting a new one).
        """
        self.current = 0
        self.num_cuts = 0
        self.longest_seen = 0

    def measure_length(self, example: Cut) -> float:
        return example.duration

    def state_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.max_duration = state_dict.pop("max_duration")
        self.max_cuts = state_dict.pop("max_cuts")
        self.current = state_dict.pop("current")
        self.num_cuts = state_dict.pop("num_cuts")
        self.longest_seen = state_dict.pop("longest_seen", 0)
        self.quadratic_duration = state_dict.pop("quadratic_duration", None)
        # backward compatibility
        state_dict.pop("strict", None)
        state_dict.pop("max_samples", None)
        state_dict.pop("max_frames", None)
        assert len(state_dict) == 0, (
            "Error in TimeConstraint.load_state_dict(): Unexpected keys:\n- "
            + "\n- ".join(state_dict.keys())
        )

    def __add__(self, other: "TimeConstraint") -> "TimeConstraint":
        for key in ("max_duration", "max_cuts", "quadratic_duration"):
            self_attr = getattr(self, key)
            other_attr = getattr(other, key)
            is_none = self_attr is None and other_attr is None
            assert is_none or isclose(self_attr, other_attr), (
                f"To add two TimeConstraint objects, they need to represent the same constraint "
                f"(got self.{key}={self_attr} != other.{key}={other_attr})."
            )
        return TimeConstraint(
            max_duration=self.max_duration,
            max_cuts=self.max_cuts,
            current=self.current + other.current,
            num_cuts=self.num_cuts + other.num_cuts,
            longest_seen=max(self.longest_seen, other.longest_seen),
            quadratic_duration=self.quadratic_duration,
        )

    def __eq__(self, other: "TimeConstraint") -> bool:
        return (
            self.max_duration == other.max_duration
            and self.max_cuts == other.max_cuts
            and self.quadratic_duration == other.quadratic_duration
        )


@dataclass
class TokenConstraint(SamplingConstraint):
    """
    Represents a token-based constraint for sampler classes that sample text data.
    It is defined as maximum total number of tokens in a mini-batch and/or max batch size.

    Similarly to :class:`TimeConstraint`, we support ``quadratic_length`` for quadratic
    token penalty when sampling longer texts.
    """

    max_tokens: int = None
    max_examples: int = None
    current: int = 0
    num_examples: int = 0
    longest_seen: int = 0
    quadratic_length: Optional[int] = None

    def __post_init__(self) -> None:
        assert is_none_or_gt(self.max_tokens, 0)
        assert is_none_or_gt(self.max_examples, 0)
        assert is_none_or_gt(self.quadratic_length, 0)

    def add(self, example: TextExample) -> None:
        """
        Increment the internal token counter for the constraint,
        selecting the right property from the input object.
        """
        if self.max_tokens is not None:
            size = self._maybe_apply_quadratic_correction(self.measure_length(example))
            self.current += size
            self.longest_seen = max(self.longest_seen, size)
        self.num_examples += 1

    def _maybe_apply_quadratic_correction(self, size: int) -> int:
        if self.quadratic_length is None:
            return size
        # For the quadratic complexity case, we add a term that accounts for
        # extra memory occupied by the model. The 1/quadratic_length term causes
        # the effective length to be doubled when it's equal to quadratic_length.
        return size + (size**2) / self.quadratic_length

    def exceeded(self) -> bool:
        """Is the constraint exceeded or not."""
        if self.max_examples is not None and self.num_examples > self.max_examples:
            return True
        if self.max_tokens is None:
            return False
        effective_duration = self.num_examples * self.longest_seen
        return effective_duration > self.max_tokens

    def close_to_exceeding(self) -> bool:
        """
        Check if the batch is close to satisfying the constraints.
        We define "closeness" as: if we added one more cut that has
        duration/num_frames/num_samples equal to the longest seen cut
        in the current batch, then the batch would have exceeded the constraints.
        """
        if self.max_examples is not None and self.num_examples >= self.max_examples:
            return True

        if self.max_tokens is not None:
            effective_size = (self.num_examples + 1) * self.longest_seen
            return effective_size > self.max_tokens
        return False

    def reset(self) -> None:
        """
        Reset the internal counter (to be used after a batch was created,
        to start collecting a new one).
        """
        self.current = 0
        self.num_examples = 0
        self.longest_seen = 0

    def measure_length(self, example: TextExample) -> float:
        return example.num_tokens


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

    def reset_current_epoch(self) -> None:
        self.stats_per_epoch[self.current_epoch] = EpochDiagnostics(self.current_epoch)

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


class _filter_nothing:
    def __call__(self, cut: Cut) -> bool:
        return True


def _and(
    fn1: Callable[[Cut], bool], fn2: Callable[[Cut], bool]
) -> Callable[[Cut], bool]:
    def _and_wrapper(cut: Cut) -> bool:
        return fn1(cut) and fn2(cut)

    return _and_wrapper
