from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Union

from lhotse import CutSet, FeatureSet, load_manifest
from lhotse.audio import RecordingSet
from lhotse.supervision import SupervisionSet
from lhotse.utils import Pathlike

DEFAULT_DETECTED_MANIFEST_TYPES = ("recordings", "supervisions")

TYPES_TO_CLASSES = {
    "recordings": RecordingSet,
    "supervisions": SupervisionSet,
    "features": FeatureSet,
    "cuts": CutSet,
}


def read_manifests_if_cached(
    dataset_parts: Optional[Sequence[str]],
    output_dir: Optional[Pathlike],
    prefix: str = "",
    suffix: Optional[str] = "jsonl.gz",
    types: Iterable[str] = DEFAULT_DETECTED_MANIFEST_TYPES,
    lazy: bool = False,
) -> Optional[Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]]:
    """
    Loads manifests from the disk, or a subset of them if only some exist.
    The manifests are searched for using the pattern ``output_dir / f'{prefix}_{manifest}_{part}.json'``,
    where `manifest` is one of ``["recordings", "supervisions"]`` and ``part`` is specified in ``dataset_parts``.
    This function is intended to speedup data preparation if it has already been done before.

    :param dataset_parts: Names of dataset pieces, e.g. in LibriSpeech: ``["test-clean", "dev-clean", ...]``.
    :param output_dir: Where to look for the files.
    :param prefix: Optional common prefix for the manifest files (underscore is automatically added).
    :param suffix: Optional common suffix for the manifest files ("json" by default).
    :param types: Which types of manifests are searched for (default: 'recordings' and 'supervisions').
    :return: A dict with manifest (``d[dataset_part]['recording'|'manifest']``) or ``None``.
    """
    if output_dir is None:
        return None
    if prefix and not prefix.endswith("_"):
        prefix = f"{prefix}_"
    if suffix.startswith("."):
        suffix = suffix[1:]
    if lazy and not suffix.startswith("jsonl"):
        raise ValueError(
            f"Only JSONL manifests can be opened lazily (got suffix: '{suffix}')"
        )
    manifests = defaultdict(dict)
    output_dir = Path(output_dir)
    for part in dataset_parts:
        for manifest in types:
            path = output_dir / f"{prefix}{manifest}_{part}.{suffix}"
            if not path.is_file():
                continue
            if lazy:
                manifests[part][manifest] = TYPES_TO_CLASSES[manifest].from_jsonl_lazy(
                    path
                )
            else:
                manifests[part][manifest] = load_manifest(path)
    return dict(manifests)


def manifests_exist(
    part: str,
    output_dir: Optional[Pathlike],
    types: Iterable[str] = DEFAULT_DETECTED_MANIFEST_TYPES,
    prefix: str = "",
    suffix: str = "json",
) -> bool:
    if output_dir is None:
        return False
    if prefix and not prefix.endswith("_"):
        prefix = f"{prefix}_"
    if suffix.startswith("."):
        suffix = suffix[1:]
    output_dir = Path(output_dir)
    for name in types:
        path = output_dir / f"{prefix}{name}_{part}.{suffix}"
        if not path.is_file():
            return False
    return True
