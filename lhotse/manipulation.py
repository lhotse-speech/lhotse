import random
from functools import reduce
from math import ceil
from operator import add
from typing import List, TypeVar, Iterable, Any

from lhotse.audio import AudioSet
from lhotse.cut import CutSet
from lhotse.features import FeatureSet
from lhotse.supervision import SupervisionSet
from lhotse.utils import Pathlike

Manifest = TypeVar('Manifest', AudioSet, SupervisionSet, FeatureSet, CutSet)


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

    if isinstance(manifest, AudioSet):
        contents = maybe_randomize(manifest.recordings.items())
        return [AudioSet(recordings=dict(contents[begin: end])) for begin, end in split_indices]

    if isinstance(manifest, SupervisionSet):
        contents = maybe_randomize(manifest.segments.items())
        return [SupervisionSet(segments=dict(contents[begin: end])) for begin, end in split_indices]

    if isinstance(manifest, FeatureSet):
        contents = maybe_randomize(manifest.features)
        return [
            FeatureSet(
                features=contents[begin: end],
                feature_extractor=manifest.feature_extractor
            )
            for begin, end in split_indices
        ]

    if isinstance(manifest, CutSet):
        contents = maybe_randomize(manifest.cuts.items())
        return [CutSet(cuts=dict(contents[begin: end])) for begin, end in split_indices]

    raise ValueError(f"Unknown type of manifest: {type(manifest)}")


def combine(*manifests: Manifest) -> Manifest:
    """Combine multiple manifests of the same type into one."""
    return reduce(add, manifests)


def load_manifest(path: Pathlike) -> Manifest:
    """Generic utility for reading an arbitrary manifest."""
    data_set = None
    for manifest_type in [AudioSet, SupervisionSet, FeatureSet, CutSet]:
        try:
            data_set = manifest_type.from_yaml(path)
        except Exception:
            pass
    if data_set is None:
        raise ValueError(f'Unknown type of manifest: {path}')
    return data_set
