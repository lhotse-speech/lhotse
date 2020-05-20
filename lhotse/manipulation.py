from math import ceil
from typing import Union, List

from lhotse.audio import AudioSet
from lhotse.features import FeatureSet
from lhotse.supervision import SupervisionSet

Manifest = Union[AudioSet, SupervisionSet, FeatureSet]


def split(manifest: Manifest, num_splits: int) -> List[Manifest]:
    num_items = len(manifest)
    if num_splits > num_items:
        raise ValueError(f"Cannot split manifest into more chunks ({num_splits}) than its number of items {num_items}")
    chunk_size = int(ceil(num_items / num_splits))
    split_indices = [(i * chunk_size, min(num_items, (i + 1) * chunk_size)) for i in range(num_splits)]

    if isinstance(manifest, AudioSet):
        contents = list(manifest.recordings.items())
        return [AudioSet(recordings=dict(contents[begin: end])) for begin, end in split_indices]

    if isinstance(manifest, SupervisionSet):
        contents = list(manifest.segments.items())
        return [SupervisionSet(segments=dict(contents[begin: end])) for begin, end in split_indices]

    if isinstance(manifest, FeatureSet):
        contents = manifest.features
        return [
            FeatureSet(
                features=contents[begin: end],
                feature_extractor=manifest.feature_extractor
            )
            for begin, end in split_indices
        ]

    raise ValueError(f"Unknown type of manifest: {type(manifest)}")
