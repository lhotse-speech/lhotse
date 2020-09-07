import random
from functools import reduce
from itertools import chain
from math import ceil
from operator import add
from typing import Any, Iterable, List, Optional, TypeVar

from lhotse.audio import Recording, RecordingSet
from lhotse.cut import Cut, CutSet, MixedCut
from lhotse.features import Features, FeatureSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, load_yaml

ManifestItem = TypeVar('ManifestItem', Recording, SupervisionSegment, Features, Cut, MixedCut)
Manifest = TypeVar('Manifest', RecordingSet, SupervisionSet, FeatureSet, CutSet)


def split(manifest: Manifest, num_splits: int, randomize: bool = False) -> List[Manifest]:
    """Split a manifest into `num_splits` equal parts. The element order can be randomized."""
    num_items = len(manifest)
    if num_splits > num_items:
        raise ValueError(f"Cannot split manifest into more chunks ({num_splits}) than its number of items {num_items}")
    chunk_size = int(ceil(num_items / num_splits))
    split_indices = [(i * chunk_size, min(num_items, (i + 1) * chunk_size)) for i in range(num_splits)]

    def maybe_randomize(items: Iterable[Any]) -> List[Any]:
        items = list(items)
        if randomize:
            random.shuffle(items)
        return items

    if isinstance(manifest, RecordingSet):
        contents = maybe_randomize(manifest.recordings.items())
        return [RecordingSet(recordings=dict(contents[begin: end])) for begin, end in split_indices]

    if isinstance(manifest, SupervisionSet):
        contents = maybe_randomize(manifest.segments.items())
        return [SupervisionSet(segments=dict(contents[begin: end])) for begin, end in split_indices]

    if isinstance(manifest, FeatureSet):
        contents = maybe_randomize(manifest.features)
        return [FeatureSet(features=contents[begin: end]) for begin, end in split_indices]

    if isinstance(manifest, CutSet):
        contents = maybe_randomize(manifest.cuts.items())
        return [CutSet(cuts=dict(contents[begin: end])) for begin, end in split_indices]

    raise ValueError(f"Unknown type of manifest: {type(manifest)}")


def combine(*manifests: Manifest) -> Manifest:
    """Combine multiple manifests of the same type into one."""
    return reduce(add, manifests)


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
    if isinstance(first_item, (Cut, MixedCut)):
        return CutSet.from_cuts(items)
    if isinstance(first_item, Features):
        raise ValueError("FeatureSet generic construction from iterable is not possible, as the config information "
                         "would have been lost. Call FeatureSet.from_features() directly instead.")

    raise ValueError(f"Unknown type of manifest item: {first_item}")


def load_manifest(path: Pathlike) -> Manifest:
    """Generic utility for reading an arbitrary manifest."""
    raw_data = load_yaml(path)
    data_set = None
    for manifest_type in [RecordingSet, SupervisionSet, FeatureSet, CutSet]:
        try:
            data_set = manifest_type.from_dicts(raw_data)
        except Exception:
            pass
    if data_set is None:
        raise ValueError(f'Unknown type of manifest: {path}')
    return data_set
