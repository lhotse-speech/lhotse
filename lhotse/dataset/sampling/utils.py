import warnings
from statistics import mean
from typing import Dict, Tuple

import numpy as np

from lhotse import CutSet
from lhotse.dataset.sampling.base import CutSampler


def find_pessimistic_batches(
    sampler: CutSampler, batch_tuple_index: int = 0
) -> Tuple[Dict[str, CutSet], Dict[str, float]]:
    """
    Function for finding 'pessimistic' batches, i.e. batches that have the highest potential
    to blow up the GPU memory during training. We will fully iterate the sampler and record
    the most risky batches under several criteria:
    - single longest cut
    - single longest supervision
    - largest batch cuts duration
    - largest batch supervisions duration
    - max num cuts
    - max num supervisions

    .. note: It is up to the users to convert the sampled CutSets into actual batches and test them
        by running forward and backward passes with their model.

    Example of how this function can be used with a PyTorch model
    and a :class:`~lhotse.dataset.K2SpeechRecognitionDataset`::

        sampler = SimpleCutSampler(cuts, max_duration=300)
        dataset = K2SpeechRecognitionDataset()
        batches, scores = find_pessimistic_batches(sampler)
        for reason, cuts in batches.items():
            try:
                batch = dset[cuts]
                outputs = model(batch)
                loss = loss_fn(outputs)
                loss.backward()
            except:
                print(f"Exception caught when evaluating pessimistic batch for: {reason}={scores[reason]}")
                raise


    :param sampler: An instance of a Lhotse :class:`.CutSampler`.
    :param batch_tuple_index: Applicable to samplers that return tuples of :class:`~lhotse.cut.CutSet`.
        Indicates which position in the tuple we should look up for the CutSet.
    :return: A tuple of dicts: the first with batches (as CutSets) and the other with criteria values, i.e.:
        ``({"<criterion>": <CutSet>, ...}, {"<criterion>": <value>, ...})``
    """
    criteria = {
        "single_longest_cut": lambda cuts: max(c.duration for c in cuts),
        "single_longest_supervision": lambda cuts: max(
            sum(s.duration for s in c.supervisions) for c in cuts
        ),
        "largest_batch_cuts_duration": lambda cuts: sum(c.duration for c in cuts),
        "largest_batch_supervisions_duration": lambda cuts: sum(
            s.duration for c in cuts for s in c.supervisions
        ),
        "max_num_cuts": len,
        "max_num_supervisions": lambda cuts: sum(
            1 for c in cuts for _ in c.supervisions
        ),
    }
    try:
        sampler = iter(sampler)
        first_batch = next(sampler)
        if isinstance(first_batch, tuple):
            first_batch = first_batch[batch_tuple_index]
    except StopIteration:
        warnings.warn("Empty sampler encountered in find_pessimistic_batches()")
        return {}, {}

    top_batches = {k: first_batch for k in criteria}
    top_values = {k: fn(first_batch) for k, fn in criteria.items()}

    for batch in sampler:
        if isinstance(batch, tuple):
            batch = batch[batch_tuple_index]
        for crit, fn in criteria.items():
            val = fn(batch)
            if val > top_values[crit]:
                top_values[crit] = val
                top_batches[crit] = batch

    return top_batches, top_values


def report_padding_ratio_estimate(sampler: CutSampler, n_samples: int = 1000) -> str:
    """
    Returns a human-readable string message about amount of padding diagnostics.
    Assumes that padding corresponds to segments without any supervision within cuts.
    """
    supervised = []
    total = []
    gaps = []
    batch_supervised = []
    batch_total = []
    batch_gaps = []
    min_dur_diffs = []
    mean_dur_diffs = []
    max_dur_diffs = []
    sampler = iter(sampler)

    for i in range(n_samples):
        try:
            batch = next(sampler)
        except StopIteration:
            break

        if not isinstance(batch, CutSet):
            warnings.warn(
                "The sampler returned a mini-batch with multiple CutSets: "
                "we will only report the padding estimate for the first CutSet in each mini-batch."
            )
            batch = batch[0]

        batch = batch.sort_by_duration(ascending=False)

        if len(batch) > 1:
            min_dur_diffs.append(
                (batch[0].duration - batch[1].duration) / batch[0].duration
            )
            max_dur_diffs.append(
                (batch[0].duration - batch[len(batch) - 1].duration) / batch[0].duration
            )
            mean_dur_diffs.append(
                mean(
                    [
                        batch[0].duration - batch[i].duration
                        for i in range(1, len(batch))
                    ]
                )
                / batch[0].duration
            )

        batch = batch.pad()
        batch_sup = 0
        batch_tot = 0
        batch_gap = 0
        for cut in batch:
            total.append(cut.duration)
            supervised.append(sum(s.duration for s in cut.supervisions))
            gaps.append(total[-1] - supervised[-1])
            batch_sup += supervised[-1]
            batch_tot += total[-1]
            batch_gap += gaps[-1]

        batch_supervised.append(batch_sup)
        batch_total.append(batch_tot)
        batch_gaps.append(batch_gap)

    m_supervised = np.mean(supervised)
    m_total = np.mean(total)
    m_gaps = np.mean(gaps)
    m_batch_supervised = np.mean(batch_supervised)
    m_batch_total = np.mean(batch_total)
    m_batch_gaps = np.mean(batch_gaps)

    return f"""An average CUT has {m_supervised:.1f}s (std={np.std(supervised):.1f}s) of supervisions vs. {m_total:.1f}s (std={np.std(total):.1f}s) of total duration. Average padding is {m_gaps:.1f}s (std={np.std(gaps):.1f}s), i.e. {m_gaps / m_total:.1%}.
An average BATCH has {m_batch_supervised:.1f}s (std={np.std(batch_supervised):.1f}s) of combined supervised duration vs. {m_batch_total:.1f}s (std={np.std(batch_total):.1f}s) of combined total duration. Average padding is {m_batch_gaps:.1f}s (std={np.std(batch_gaps):.1f}s), i.e. {m_batch_gaps / m_batch_total:.1%}.
Expected variability of cut durations within a single batch is +/-{np.mean(mean_dur_diffs):.1%} (two closest cuts: {np.mean(min_dur_diffs):.1%}, two most distant cuts: {np.mean(max_dur_diffs):.1%}).
    """
