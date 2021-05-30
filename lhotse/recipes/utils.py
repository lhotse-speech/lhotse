from collections import defaultdict
from typing import Dict, Optional, Sequence, Union

from lhotse import load_manifest
from lhotse.audio import RecordingSet
from lhotse.supervision import SupervisionSet
from lhotse.utils import Pathlike


def read_manifests_if_cached(
        dataset_parts: Optional[Sequence[str]],
        output_dir: Optional[Pathlike],
        prefix: str = '',
        suffix: Optional[str] = 'json'
) -> Optional[Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]]:
    """
    Loads manifests from the disk if all of them exist in the specified paths.
    the manifests are searched for using the pattern `output_dir / f'{prefix}_{manifest}_{part}.json'`,
    where `manifest` is one of `["recordings", "supervisions"]` and `part` is specified in `dataset_parts`.
    This function is intended to speedup data preparation if it has already been done before.

    :param dataset_parts: Names of dataset pieces, e.g. in LibriSpeech: ``["test-clean", "dev-clean", ...]``.
    :param output_dir: Where to look for the files.
    :param prefix: Optional common prefix for the manifest files (underscore is automatically added).
    :param suffix: Optional common suffix for the manifest files ("json" by default).
    :return: A dict with manifest (``d[dataset_part]['recording'|'manifest']``) or ``None``.
    """
    if output_dir is None:
        return None
    if prefix:
        prefix = f'{prefix}_'
    manifests = defaultdict(dict)
    for part in dataset_parts:
        for manifest in ('recordings', 'supervisions'):
            path = output_dir / f'{prefix}{manifest}_{part}.{suffix}'
            if not path.is_file():
                # If one of the manifests is not available, assume we need to read and prepare everything
                # to simplify the rest of the code.
                return None
            manifests[part][manifest] = load_manifest(path)
    return dict(manifests)
