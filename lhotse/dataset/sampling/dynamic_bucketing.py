import random
import warnings
from bisect import bisect_right
from collections import deque
from typing import Deque, Generator, Iterable, List, Optional

import numpy as np

from lhotse import CutSet, Seconds
from lhotse.cut import Cut
from lhotse.dataset.sampling.base import TimeConstraint


def estimate_duration_buckets(cuts: Iterable[Cut], num_buckets: int) -> List[Seconds]:
    """
    Given an iterable of cuts and a desired number of buckets, select duration values
    that should start each bucket.

    The returned list, ``bins``, has ``num_buckets - 1`` elements.
    The first bucket should contain cuts with duration ``0 <= d < bins[0]``;
    the last bucket should contain cuts with duration ``bins[-1] <= d < float("inf")``,
    ``i``-th bucket should contain cuts with duration ``bins[i - 1] <= d < bins[i]``.

    :param cuts: an iterable of :class:`lhotse.cut.Cut`.
    :param num_buckets: desired number of buckets.
    :return: a list of boundary duration values (floats).
    """
    assert num_buckets > 1

    durs = np.array([c.duration for c in cuts])
    durs.sort()
    assert num_buckets < durs.shape[0], (
        f"The number of buckets ({num_buckets}) must be smaller "
        f"than the number of cuts ({durs.shape[0]})."
    )
    bucket_duration = durs.sum() / num_buckets

    bins = []
    tot = 0.0
    for dur in durs:
        if tot > bucket_duration:
            bins.append(dur)
            tot = 0.0
        tot += dur

    return bins


def dynamic_bucketing(
    cuts: Iterable[Cut],
    duration_bins: List[Seconds],
    max_duration: float,
    drop_last: bool = False,
    buffer_size: int = 10000,
    rng: random.Random = None,
) -> Generator[CutSet, None, None]:

    if rng is None:
        rng = random.Random()

    assert duration_bins == sorted(duration_bins), (
        f"Argument list for 'duration_bins' is expected to be in "
        f"sorted order (got: {duration_bins})."
    )

    # A heuristic diagnostic first, for finding the right settings.
    mean_duration = np.mean(duration_bins)
    expected_buffer_duration = buffer_size * mean_duration
    expected_bucket_duration = expected_buffer_duration / (len(duration_bins) + 1)
    if expected_bucket_duration < max_duration:
        warnings.warn(
            f"Your 'buffer_size' setting of {buffer_size} might be too low to satisfy "
            f"a 'max_duration' of {max_duration} (given our best guess)."
        )

    # Init: create empty buckets (note: `num_buckets = len(duration_bins) + 1`).
    buckets = [deque() for _ in range(len(duration_bins) + 1)]

    # Init: sample `buffer_size` cuts and assign them to the right buckets.
    cuts_iter = iter(cuts)

    def collect_cuts_in_buckets(n_cuts: int):
        try:
            for _ in range(n_cuts):
                cut = next(cuts_iter)
                bucket_idx = bisect_right(duration_bins, cut.duration)
                buckets[bucket_idx].append(cut)
        except StopIteration:
            pass

    collect_cuts_in_buckets(buffer_size)

    # Init: determine which buckets are "ready"
    def is_ready(bucket: Deque[Cut]):
        tot = TimeConstraint(max_duration=max_duration)
        for c in bucket:
            tot.add(c)
            if tot.close_to_exceeding():
                return True
        return False

    assert any(is_ready(bucket) for bucket in buckets)

    # The iteration code starts here.
    # On each step we're sampling a new batch.
    try:
        while True:
            ready_buckets = [b for b in buckets if is_ready(b)]
            if not ready_buckets:
                # No bucket has enough data to yield for the last full batch.
                non_empty_buckets = [b for b in buckets if b]
                if drop_last or len(non_empty_buckets) == 0:
                    # Either the user requested only full batches, or we have nothing left.
                    raise StopIteration()
                else:
                    # Sample from partial batches that are left.
                    ready_buckets = non_empty_buckets
            # Choose a bucket to sample from.
            # We'll only select from the buckets that have a full batch available.
            sampling_bucket = rng.choice(ready_buckets)
            # Sample one batch from that bucket and yield it to the caller.
            batcher = DurationBatcher(sampling_bucket, max_duration=max_duration)
            batch = next(iter(batcher))
            batch_size = len(batch)
            yield batch
            # Remove sampled cuts from the bucket.
            for _ in range(batch_size):
                sampling_bucket.popleft()
            # Fetch new cuts and add them to appropriate buckets.
            collect_cuts_in_buckets(batch_size)
    except StopIteration:
        pass


# Note: this class is a subset of SingleCutSampler and is "datapipes" ready.
class DurationBatcher:
    def __init__(
        self,
        datapipe,
        max_frames: int = None,
        max_samples: int = None,
        max_duration: Seconds = None,
        max_cuts: Optional[int] = None,
        drop_last: bool = False,
    ):
        from lhotse.dataset.sampling.base import SamplingDiagnostics, TimeConstraint

        self.datapipe = datapipe
        self.reuse_cuts_buffer = deque()
        self.drop_last = drop_last
        self.max_cuts = max_cuts
        self.diagnostics = SamplingDiagnostics()
        self.time_constraint = TimeConstraint(
            max_duration=max_duration, max_frames=max_frames, max_samples=max_samples
        )

    def __iter__(self):
        self.cuts_iter = iter(self.datapipe)
        try:
            while True:
                yield self._collect_batch()
        except StopIteration:
            pass
        self.cuts_iter = None

    def _collect_batch(self):
        self.time_constraint.reset()
        cuts = []
        while True:
            # Check that we have not reached the end of the dataset.
            try:
                if self.reuse_cuts_buffer:
                    next_cut = self.reuse_cuts_buffer.popleft()
                else:
                    # If this doesn't raise (typical case), it's not the end: keep processing.
                    next_cut = next(self.cuts_iter)
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
                    self.reuse_cuts_buffer.append(next_cut)
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
