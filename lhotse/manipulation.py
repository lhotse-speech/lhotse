from concurrent.futures import ProcessPoolExecutor
from functools import reduce
from itertools import chain
from operator import add
from typing import Callable, Iterable, Optional, TypeVar, Union

from lhotse.audio import Recording, RecordingSet
from lhotse.cut import CutSet, MixedCut, MonoCut
from lhotse.features import FeatureSet, Features
from lhotse.supervision import SupervisionSegment, SupervisionSet

ManifestItem = TypeVar(
    "ManifestItem", Recording, SupervisionSegment, Features, MonoCut, MixedCut
)
Manifest = TypeVar("Manifest", RecordingSet, SupervisionSet, FeatureSet, CutSet)


def combine(*manifests: Union[Manifest, Iterable[Manifest]]) -> Manifest:
    """
    Combine multiple manifests of the same type into one.

    Examples:
        >>> # Pass several arguments
        >>> combine(recording_set1, recording_set2, recording_set3)
        >>> # Or pass a single list/tuple of manifests
        >>> combine([supervision_set1, supervision_set2])
    """
    if len(manifests) == 1 and isinstance(manifests, (tuple, list)):
        manifests = manifests[0]
    return reduce(add, manifests)


def split_parallelize_combine(
    num_jobs: int, manifest: Manifest, fn: Callable, *args, **kwargs
) -> Manifest:
    """
    Convenience wrapper that parallelizes the execution of functions
    that transform manifests. It splits the manifests into ``num_jobs``
    pieces, applies the function to each split, and then combines the splits.

    This function is used internally in Lhotse to implement some parallel ops.

    .. hint:
        This way of parallel execution is the most optimal when each item
        in the manifest is being processed with "simple" operations that take
        a short amount of time -- a ``ProcessPoolExecutor`` would have added
        a lot of overhead for sending individual items over IPC.

    Example::

        >>> from lhotse import CutSet, split_parallelize_combine
        >>> cuts = CutSet(...)
        >>> window_cuts = split_parallelize_combine(
        ...     16,
        ...     cuts,
        ...     CutSet.cut_into_windows,
        ...     duration=30.0
        ... )

    :param num_jobs: The number of parallel jobs.
    :param manifest: The manifest to be processed.
    :param fn: Function or method that transforms the manifest; the first parameter
        has to be ``manifest`` (for methods, they have to be methods on that manifests type,
        e.g. ``CutSet.cut_into_windows``.
    :param args: positional arguments to ``fn``.
    :param kwargs keyword arguments to ``fn``.
    """
    splits = manifest.split(num_splits=num_jobs)
    with ProcessPoolExecutor(num_jobs) as ex:
        futures = [ex.submit(fn, subset, *args, **kwargs) for subset in splits]
        result = combine([f.result() for f in futures])
    return result


def to_manifest(items: Iterable[ManifestItem]) -> Optional[Manifest]:
    """
    Take an iterable of data types in Lhotse such as Recording, SupervisonSegment or Cut, and create the manifest of the
    corresponding type. When the iterable is empty, returns None.
    """
    items = iter(items)
    try:
        first_item = next(items)
    except StopIteration:
        return None
    items = chain([first_item], items)

    if isinstance(first_item, Recording):
        return RecordingSet.from_recordings(items)
    if isinstance(first_item, SupervisionSegment):
        return SupervisionSet.from_segments(items)
    if isinstance(first_item, (MonoCut, MixedCut)):
        return CutSet.from_cuts(items)
    if isinstance(first_item, Features):
        raise ValueError(
            "FeatureSet generic construction from iterable is not possible, as the config information "
            "would have been lost. Call FeatureSet.from_features() directly instead."
        )

    raise ValueError(f"Unknown type of manifest item: {first_item}")
