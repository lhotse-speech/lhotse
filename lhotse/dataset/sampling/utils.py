import warnings
from typing import Dict, Tuple

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

        sampler = SingleCutSampler(cuts, max_duration=300)
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
