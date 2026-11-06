"""
Packed sequence sampling: dynamic samplers that fill each mini-batch with as many tokens
as possible without exceeding an exact token budget, for models that consume
mini-batches as packed sequences (concatenated without padding).
"""

import math
from dataclasses import dataclass, field
from itertools import islice
from queue import Queue
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from lhotse import CutSet
from lhotse.cut import Cut
from lhotse.dataset.sampling.base import SamplingConstraint
from lhotse.dataset.sampling.checkpoint_backends import _all_sources_graph_restorable
from lhotse.dataset.sampling.dynamic import (
    DurationBatcher,
    DynamicCutSampler,
    detuplify,
)
from lhotse.dataset.sampling.dynamic_bucketing import (
    DynamicBucketer,
    DynamicBucketingSampler,
)
from lhotse.lazy import require_graph_origin, resolve_iterator_source

Example = Union[Cut, Tuple[Cut, ...]]


@dataclass
class PackedTokenConstraint(SamplingConstraint):
    """
    Token constraint for mini-batches that are packed (concatenated without padding)
    through the model, e.g. with variable-length attention kernels.

    :class:`~lhotse.dataset.sampling.base.TokenConstraint` budgets padded work as
    the number of examples times the longest example. For packed execution, compute
    and activation memory scale with the sum of example lengths instead, so
    ``batch_tokens`` is a hard cap on the total number of tokens in a mini-batch.
    Use it with :class:`PackedSequenceDynamicCutSampler` or
    :class:`PackedSequenceDynamicBucketingSampler`, which select examples that fill
    ``batch_tokens`` as closely as possible.

    The length of an example is read from its ``num_tokens`` attribute, and its
    output length (for ``batch_output_tokens``) from ``num_output_tokens``;
    override :meth:`measure_length` and :meth:`measure_output_length` to measure
    them differently. Lengths must be positive integers.

    :param batch_tokens: The maximum total number of tokens in a mini-batch.
    :param max_examples: The maximum number of examples in a mini-batch.
    :param quadratic_length: When set, an example of length ``L`` takes
        ``ceil(L + L**2 / quadratic_length)`` tokens of ``batch_tokens``,
        to account for models with quadratic complexity.
    :param batch_output_tokens: The maximum total number of output tokens in a mini-batch.
    :param bucket_bins: Sorted bucket boundaries in tokens, used with ``max_examples_per_bucket``.
    :param max_examples_per_bucket: The maximum number of examples in a mini-batch
        whose first example falls into a given bucket of ``bucket_bins``.
        Examples longer than the last boundary are rejected.
    """

    batch_tokens: int
    max_examples: Optional[int] = None
    quadratic_length: Optional[float] = None
    batch_output_tokens: Optional[int] = None
    bucket_bins: Optional[Sequence[int]] = None
    max_examples_per_bucket: Optional[Sequence[int]] = None
    current: int = 0
    current_output: int = 0
    num_examples: int = 0
    _batch_max_examples: Optional[int] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        for name in (
            "batch_tokens",
            "max_examples",
            "quadratic_length",
            "batch_output_tokens",
        ):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive (got {value})")
        if self.batch_tokens is None:
            raise ValueError("PackedTokenConstraint requires batch_tokens.")
        if (self.bucket_bins is None) != (self.max_examples_per_bucket is None):
            raise ValueError(
                "bucket_bins and max_examples_per_bucket must be set together."
            )
        if self.bucket_bins is not None:
            if list(self.bucket_bins) != sorted(self.bucket_bins):
                raise ValueError("bucket_bins must be sorted in ascending order.")
            if len(self.bucket_bins) != len(self.max_examples_per_bucket):
                raise ValueError(
                    "bucket_bins and max_examples_per_bucket must have equal lengths."
                )
            if any(size <= 0 for size in self.max_examples_per_bucket):
                raise ValueError("max_examples_per_bucket values must be positive.")

    def measure_length(self, example: Any) -> int:
        return example.num_tokens

    def measure_output_length(self, example: Any) -> int:
        return example.num_output_tokens

    def measure_packing_length(self, example: Any) -> int:
        """Return how many tokens of ``batch_tokens`` the example takes."""
        size = self.measure_length(example)
        if self.quadratic_length is None:
            return math.ceil(size)
        return math.ceil(size + size**2 / self.quadratic_length)

    def max_examples_for_length(self, num_tokens: int) -> Optional[int]:
        """Return the maximum number of examples in a mini-batch whose first example has ``num_tokens``."""
        if self.bucket_bins is None:
            return self.max_examples
        bucket_idx = self.select_bucket(self.bucket_bins, example_len=num_tokens)
        if bucket_idx >= len(self.bucket_bins):
            raise ValueError(
                f"Example length {num_tokens} exceeds the highest bucket boundary {self.bucket_bins[-1]}."
            )
        bucket_limit = self.max_examples_per_bucket[bucket_idx]
        if self.max_examples is None:
            return bucket_limit
        return min(self.max_examples, bucket_limit)

    def add(self, example: Any) -> None:
        if self.num_examples == 0:
            self._batch_max_examples = self.max_examples_for_length(
                self.measure_length(example)
            )
        self.current += self.measure_packing_length(example)
        if self.batch_output_tokens is not None:
            self.current_output += self.measure_output_length(example)
        self.num_examples += 1

    def exceeded(self) -> bool:
        if (
            self._batch_max_examples is not None
            and self.num_examples > self._batch_max_examples
        ):
            return True
        if (
            self.batch_output_tokens is not None
            and self.current_output > self.batch_output_tokens
        ):
            return True
        return self.current > self.batch_tokens

    def close_to_exceeding(self) -> bool:
        """
        Packed samplers enforce the limits exactly when selecting examples;
        this is only a fullness heuristic: would one more example of the mean length exceed a limit?
        """
        if (
            self._batch_max_examples is not None
            and self.num_examples >= self._batch_max_examples
        ):
            return True
        if self.num_examples == 0:
            return False
        if self.batch_output_tokens is not None:
            mean_output = self.current_output / self.num_examples
            if self.current_output + mean_output > self.batch_output_tokens:
                return True
        mean_length = self.current / self.num_examples
        return self.current + mean_length > self.batch_tokens

    def reset(self) -> None:
        self.current = 0
        self.current_output = 0
        self.num_examples = 0
        self._batch_max_examples = None


class PackedSequenceDynamicCutSampler(DynamicCutSampler):
    """
    A variant of :class:`~lhotse.dataset.sampling.DynamicCutSampler` that packs
    examples into mini-batches up to an exact token budget.

    It keeps a pool of ``packing_buffer_size`` examples. The oldest example in the pool
    (the anchor) always goes into the next mini-batch, which guarantees that no example
    is starved; the rest of the mini-batch is the subset of the pool that fills
    ``constraint.batch_tokens`` best, preferring earlier examples on ties.

    The ``constraint`` is a :class:`PackedTokenConstraint` (or another constraint
    implementing its interface). Example::

        >>> sampler = PackedSequenceDynamicCutSampler(
        ...     examples, constraint=PackedTokenConstraint(batch_tokens=8192)
        ... )

    With graph-restorable (indexed) sources, the pool is checkpointed as graph tokens
    and restored in constant time; otherwise it is rebuilt by replay.
    """

    def __init__(self, *cuts: Iterable, packing_buffer_size: int = 128, **kwargs):
        """
        :param cuts: one or more CutSets (when more than one, will yield tuples of CutSets as mini-batches)
        :param packing_buffer_size: The number of examples considered when packing a mini-batch.
        :param kwargs: Arguments of :class:`~lhotse.dataset.sampling.DynamicCutSampler`.
        """
        super().__init__(*cuts, **kwargs)
        _validate_packing_setup(self.constraint, packing_buffer_size)
        self.packing_buffer_size = packing_buffer_size
        self._batcher = None
        self._restored_packing_buffer_tokens = None

    def state_dict(self) -> Dict[str, Any]:
        sd = super().state_dict()
        sd["packing_buffer_size"] = self.packing_buffer_size
        if not _all_sources_graph_restorable(self):
            # Replay restoration rebuilds the pool deterministically.
            sd["packing_buffer_tokens"] = None
        elif self._batcher is None:
            sd["packing_buffer_tokens"] = self._restored_packing_buffer_tokens
        else:
            sd["packing_buffer_tokens"] = [
                tuple(
                    require_graph_origin(
                        cut, "PackedSequenceDynamicCutSampler checkpoint"
                    )
                    for cut in example
                )
                for example in self._batcher.reuse_cuts_buffer
            ]
        return sd

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        self.packing_buffer_size = sd.pop("packing_buffer_size")
        self._restored_packing_buffer_tokens = sd.pop("packing_buffer_tokens")
        self._batcher = None
        super().load_state_dict(sd)

    def allow_iter_to_reset_state(self):
        super().allow_iter_to_reset_state()
        self._restored_packing_buffer_tokens = None

    def _create_batcher(self, examples: Iterable) -> "PackedSequenceBatcher":
        self._batcher = PackedSequenceBatcher(
            examples,
            constraint=self.constraint,
            drop_last=self.drop_last,
            diagnostics=self.diagnostics,
            packing_buffer_size=self.packing_buffer_size,
        )
        # Indexed restore resumes each source after the pooled examples,
        # so we put them back into the pool.
        for tokens in self._restored_packing_buffer_tokens or ():
            self._batcher.reuse_cuts_buffer.append(
                tuple(
                    resolve_iterator_source(source)[token]
                    for source, token in zip(self._active_cuts, tokens)
                )
            )
        self._restored_packing_buffer_tokens = None
        return self._batcher


class PackedSequenceDynamicBucketingSampler(DynamicBucketingSampler):
    """
    A variant of :class:`~lhotse.dataset.sampling.DynamicBucketingSampler` that packs
    examples into mini-batches up to an exact token budget.

    Each mini-batch is packed from the first ``packing_buffer_size`` examples of
    the selected bucket (or, with ``shuffle=True``, the oldest example and a random
    sample of the rest). The oldest example (the anchor) always goes into the
    mini-batch, which guarantees that no example is starved; the rest of the
    mini-batch is the subset of candidates that fills ``constraint.batch_tokens``
    best, preferring earlier examples on ties. ``buffer_size`` still bounds the
    total number of examples held across all buckets.

    The ``constraint`` is a :class:`PackedTokenConstraint` (or another constraint
    implementing its interface), and ``duration_bins`` are expressed in tokens. Example::

        >>> sampler = PackedSequenceDynamicBucketingSampler(
        ...     examples,
        ...     constraint=PackedTokenConstraint(batch_tokens=8192),
        ...     duration_bins=[256, 512, 1024],
        ... )
    """

    def __init__(self, *cuts: Iterable, packing_buffer_size: int = 128, **kwargs):
        """
        :param cuts: one or more CutSets (when more than one, will yield tuples of CutSets as mini-batches)
        :param packing_buffer_size: The number of examples considered when packing a mini-batch.
        :param kwargs: Arguments of :class:`~lhotse.dataset.sampling.DynamicBucketingSampler`.
        """
        super().__init__(*cuts, **kwargs)
        _validate_packing_setup(self.constraint, packing_buffer_size)
        self.packing_buffer_size = packing_buffer_size

    def state_dict(self) -> Dict[str, Any]:
        sd = super().state_dict()
        sd["packing_buffer_size"] = self.packing_buffer_size
        return sd

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        self.packing_buffer_size = sd.pop("packing_buffer_size")
        super().load_state_dict(sd)

    def _create_bucketer(self, cuts: Iterable, **kwargs) -> "PackedSequenceBucketer":
        return PackedSequenceBucketer(
            cuts, packing_buffer_size=self.packing_buffer_size, **kwargs
        )


class PackedSequenceBatcher(DurationBatcher):
    """Packs mini-batches from a pool of the next ``packing_buffer_size`` examples."""

    def __init__(self, *args, packing_buffer_size: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.packing_buffer_size = packing_buffer_size
        self._source_exhausted = False

    def _collect_batch(self) -> Union[CutSet, Tuple[CutSet]]:
        pool = self.reuse_cuts_buffer
        while len(pool) < self.packing_buffer_size and not self._source_exhausted:
            try:
                pool.append(next(self.cuts_iter))
            except StopIteration:
                self._source_exhausted = True
        if not pool:
            raise StopIteration()

        selected = set(_select_packed_batch(self.constraint, list(pool)))
        examples = [example for idx, example in enumerate(pool) if idx in selected]
        deferred = [example for idx, example in enumerate(pool) if idx not in selected]
        pool.clear()
        pool.extend(deferred)

        if (
            self.drop_last
            and self._source_exhausted
            and not pool
            and not _is_full_batch(self.constraint, examples)
        ):
            self.diagnostics.discard(_first(example) for example in examples)
            raise StopIteration()
        return detuplify(examples)


class PackedSequenceBucketer(DynamicBucketer):
    """Packs each mini-batch from at most ``packing_buffer_size`` examples of the selected bucket."""

    def __init__(self, *args, packing_buffer_size: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.packing_buffer_size = packing_buffer_size

    def _plan_batch(
        self, bucket: Queue, randomize_tail: bool
    ) -> Tuple[List[int], List[Example]]:
        with bucket.mutex:
            if randomize_tail:
                # The anchor stays first; the rest of the candidates is a random sample of the bucket.
                indexes = [
                    0,
                    *self.rng.sample(
                        range(1, len(bucket.queue)),
                        k=min(self.packing_buffer_size, len(bucket.queue)) - 1,
                    ),
                ]
                candidates = [bucket.queue[idx] for idx in indexes]
            else:
                candidates = list(islice(bucket.queue, self.packing_buffer_size))
                indexes = list(range(len(candidates)))
        selected = _select_packed_batch(self.constraint, candidates)
        return [indexes[pos] for pos in selected], [candidates[pos] for pos in selected]

    def _is_ready(self, bucket: Queue) -> bool:
        if bucket.qsize() == 0:
            return False
        _, examples = self._plan_batch(bucket, randomize_tail=False)
        return _is_full_batch(self.constraint, examples)

    def _collect_batch(
        self, bucket: Queue
    ) -> Tuple[Union[CutSet, Tuple[CutSet]], List[int]]:
        indexes, examples = self._plan_batch(bucket, randomize_tail=self.shuffle)
        if self.shuffle and not _is_full_batch(self.constraint, examples):
            # The bucket was selected because its prefix packs a full batch;
            # use it rather than a sparser random sample.
            prefix_indexes, prefix_examples = self._plan_batch(
                bucket, randomize_tail=False
            )
            if _is_full_batch(self.constraint, prefix_examples):
                indexes, examples = prefix_indexes, prefix_examples
        return detuplify(examples), indexes


def _validate_packing_setup(
    constraint: Optional[SamplingConstraint], packing_buffer_size: int
) -> None:
    if not hasattr(constraint, "measure_packing_length"):
        raise ValueError(
            "Packed sequence samplers require constraint=PackedTokenConstraint(...) "
            f"or a constraint implementing its interface (got {constraint!r})."
        )
    if packing_buffer_size <= 0:
        raise ValueError(
            f"packing_buffer_size must be positive (got {packing_buffer_size})."
        )


def _first(example: Example) -> Cut:
    """With multiple input CutSets, the constraint measures only the cut from the first one."""
    return example[0] if isinstance(example, tuple) else example


def _is_full_batch(constraint: SamplingConstraint, examples: List[Example]) -> bool:
    constraint = constraint.copy()
    constraint.reset()
    for example in examples:
        constraint.add(_first(example))
    assert not constraint.exceeded(), "A packed batch exceeded its constraint."
    return constraint.close_to_exceeding()


def _select_packed_batch(
    constraint: SamplingConstraint, candidates: List[Example]
) -> List[int]:
    """
    Return the sorted positions of ``candidates`` that form the next packed mini-batch.
    The first candidate (the anchor) is always selected.
    """
    examples = [_first(example) for example in candidates]
    for example in examples:
        length = constraint.measure_length(example)
        if length != int(length) or length <= 0:
            raise ValueError(
                f"Packed sequence sampling requires positive integer token lengths, but measured {length!r}."
            )
    batch_tokens = constraint.batch_tokens
    anchor_length = int(constraint.measure_length(examples[0]))
    if anchor_length > batch_tokens:
        raise ValueError(
            f"An individual example ({anchor_length} tokens) exceeds batch_tokens={batch_tokens}. "
            f"Filter out examples longer than batch_tokens before sampling."
        )
    budgets = [constraint.measure_packing_length(example) for example in examples]
    if budgets[0] > batch_tokens:
        raise ValueError(
            f"An individual example's effective packed budget ({budgets[0]} tokens) exceeds "
            f"batch_tokens={batch_tokens}; increase batch_tokens or quadratic_length."
        )
    max_examples = constraint.max_examples_for_length(anchor_length)
    max_items = None if max_examples is None else max_examples - 1
    tail = _select_best_fit_indices(
        budgets[1:], batch_tokens - budgets[0], max_items=max_items
    )
    if constraint.batch_output_tokens is not None:
        tail = _fit_output_cap(
            tail,
            budgets,
            [constraint.measure_output_length(example) for example in examples],
            batch_tokens=batch_tokens,
            output_cap=constraint.batch_output_tokens,
            max_items=max_items,
        )
    return [0, *(idx + 1 for idx in tail)]


def _select_best_fit_indices(
    lengths: Sequence[int], capacity: int, max_items: Optional[int] = None
) -> List[int]:
    """
    Select the subset of ``lengths`` with the largest sum not exceeding ``capacity``
    and at most ``max_items`` items, preferring fewer and then earlier items on ties.
    """
    if not lengths or capacity == 0 or max_items == 0:
        return []

    # Bit ``s`` of a reachability mask is set when some subset sums to ``s``.
    mask = (1 << (capacity + 1)) - 1

    if max_items is None or max_items >= len(lengths):
        # prefixes[i] holds the sums reachable with the first i items.
        # Reconstruction skips an item when its target was reachable without it,
        # so equal-quality solutions favor earlier items.
        prefixes = [1]
        for length in lengths:
            reachable = prefixes[-1]
            prefixes.append(reachable | ((reachable << length) & mask))

        target = prefixes[-1].bit_length() - 1
        selected = []
        for idx in range(len(lengths) - 1, -1, -1):
            if (prefixes[idx] >> target) & 1:
                continue
            selected.append(idx)
            target -= lengths[idx]
        selected.reverse()
        return selected

    # With an item limit, track reachable sums separately for each item count.
    layers = [1] + [0] * max_items
    prefixes = [tuple(layers)]
    for item_idx, length in enumerate(lengths):
        for count in range(min(max_items, item_idx + 1), 0, -1):
            layers[count] |= (layers[count - 1] << length) & mask
        prefixes.append(tuple(layers))

    reachable = 0
    for layer in layers:
        reachable |= layer
    target = reachable.bit_length() - 1
    count = next(count for count, layer in enumerate(layers) if (layer >> target) & 1)
    selected = []
    for idx in range(len(lengths) - 1, -1, -1):
        if (prefixes[idx][count] >> target) & 1:
            continue
        selected.append(idx)
        target -= lengths[idx]
        count -= 1
    selected.reverse()
    return selected


def _fit_output_cap(
    tail: Sequence[int],
    budget_lengths: Sequence[int],
    output_lengths: Sequence[int],
    *,
    batch_tokens: int,
    output_cap: int,
    max_items: Optional[int],
) -> List[int]:
    """
    Make a best-fit selection also respect the output-token cap.

    ``budget_lengths`` and ``output_lengths`` cover all candidates with the anchor at index 0;
    ``tail`` indexes the candidates without the anchor, as :func:`_select_best_fit_indices` returns it.
    The selected items with the largest output lengths are dropped until the cap holds,
    then the remaining candidates are backfilled first-fit under both limits.
    """
    if output_lengths[0] > output_cap:
        raise ValueError(
            f"An individual example's output tokens ({output_lengths[0]}) exceed "
            f"batch_output_tokens={output_cap}; filter it out or raise the cap."
        )
    budget_capacity = batch_tokens - budget_lengths[0]
    output_capacity = output_cap - output_lengths[0]
    budget_lengths = budget_lengths[1:]
    output_lengths = output_lengths[1:]
    kept = list(tail)
    total_output = sum(output_lengths[idx] for idx in kept)
    while kept and total_output > output_capacity:
        # On ties, the later candidate goes, so the selection keeps favoring earlier examples.
        largest = max(reversed(kept), key=lambda idx: output_lengths[idx])
        kept.remove(largest)
        total_output -= output_lengths[largest]
    chosen = set(kept)
    total_budget = sum(budget_lengths[idx] for idx in kept)
    for idx, (budget, output) in enumerate(zip(budget_lengths, output_lengths)):
        if idx in chosen:
            continue
        if max_items is not None and len(kept) >= max_items:
            break
        if (
            total_budget + budget <= budget_capacity
            and total_output + output <= output_capacity
        ):
            kept.append(idx)
            chosen.add(idx)
            total_budget += budget
            total_output += output
    return sorted(kept)
