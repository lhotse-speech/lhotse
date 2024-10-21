"""
Description taken from the abstract of the paper:
"EARS: An Anechoic Fullband Speech Dataset Benchmarked for Speech Enhancement and Dereverberation"
https://arxiv.org/abs/2406.06185

We release the EARS (Expressive Anechoic Recordings of Speech) dataset, a high-quality speech dataset comprising
107 speakers from diverse backgrounds, totaling in 100 hours of clean, anechoic speech data. The dataset covers
a large range of different speaking styles, including emotional speech, different reading styles, non-verbal sounds,
and conversational freeform speech. We benchmark various methods for speech enhancement and dereverberation on the
dataset and evaluate their performance through a set of instrumental metrics. In addition, we conduct a listening
test with 20 participants for the speech enhancement task, where a generative method is preferred. We introduce
a blind test set that allows for automatic online evaluation of uploaded data. Dataset download links and automatic
evaluation server can be found online.
"""


import json
import logging
import re
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Union

from tqdm import tqdm

from lhotse import (
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
    fix_manifests,
    validate_recordings_and_supervisions,
)
from lhotse.recipes.utils import (
    DEFAULT_DETECTED_MANIFEST_TYPES,
    TYPES_TO_CLASSES,
    load_manifest,
    manifests_exist,
)
from lhotse.utils import Pathlike, resumable_download


def _read_manifests_if_cached_no_parts(
    output_dir: Optional[Pathlike],
    prefix: str = "",
    suffix: str = "jsonl.gz",
    types: Iterable[str] = DEFAULT_DETECTED_MANIFEST_TYPES,
    lazy: bool = False,
) -> Optional[Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]]:
    """
    Loads manifests from the disk, or a subset of them if only some exist.
    The manifests are searched for using the pattern ``output_dir / f'{prefix}_{manifest}_{part}.json'``,
    where `manifest` is one of ``["recordings", "supervisions"]`` and ``part`` is specified in ``dataset_parts``.
    This function is intended to speedup data preparation if it has already been done before.

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
    for manifest in types:
        path = output_dir / f"{prefix}{manifest}.{suffix}"
        if not path.is_file():
            continue
        if lazy:
            manifests[manifest] = TYPES_TO_CLASSES[manifest].from_jsonl_lazy(path)
        else:
            manifests[manifest] = load_manifest(path)
    return dict(manifests)


def download_ears(
    target_dir: Pathlike = ".",
    force_download: bool = False,
) -> Path:
    """
    Download and unzip the EARS dataset.

    :param target_dir: Pathlike, the path of the dir to storage the dataset.
    :param force_download: Bool, if True, download the tars no matter if the tars exist.
    :return: the path to downloaded and extracted directory with data.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    resumable_download(
        "https://raw.githubusercontent.com/facebookresearch/ears_dataset/main/speaker_statistics.json",
        filename=target_dir / "speaker_statistics.json",
        force_download=force_download,
    )
    resumable_download(
        "https://raw.githubusercontent.com/facebookresearch/ears_dataset/main/transcripts.json",
        filename=target_dir / "transcripts.json",
        force_download=force_download,
    )
    for part in tqdm(
        range(1, 108), desc="Downloading the 107 speakers of the EARS dataset"
    ):
        part = f"p{part:03d}"
        url = f"https://github.com/facebookresearch/ears_dataset/releases/download/dataset"
        zip_name = f"{part}.zip"
        zip_path = target_dir / zip_name
        part_dir = target_dir / part
        completed_detector = part_dir / ".completed"
        if completed_detector.is_file():
            logging.info(f"Skipping {part} because {completed_detector} exists.")
            continue
        full_url = f"{url}/{zip_name}"
        resumable_download(full_url, filename=zip_path, force_download=force_download)
        shutil.rmtree(part_dir, ignore_errors=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(path=target_dir)
        completed_detector.touch()

    return target_dir


def prepare_ears(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    num_jobs: int = 1,
) -> Dict[str, Union[RecordingSet, SupervisionSet]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions.
    When all the manifests are available in the ``output_dir``, it will simply read and return them.

    :param corpus_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param num_jobs: the number of parallel workers parsing the data.
    :return: a Dict whose keys are 'recordings' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    dataset_parts = [f"p{spk:03d}" for spk in range(1, 108)]
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Maybe the manifests already exist: we can read them and save a bit of preparation time.
        manifests = _read_manifests_if_cached_no_parts(
            output_dir=output_dir, prefix="ears"
        )

    # Contents of the file
    # {
    #     "p001": {
    #         "age": "36-45",
    #         "ethnicity": "white or caucasian",
    #         "gender": "male",
    #         "weight": "160 - 180 lbs",
    #         "native language": "german",
    #         "height": "6' - 6'3"
    #     },
    #     ...
    # }
    spk2meta = json.loads((corpus_dir / "speaker_statistics.json").read_text())

    # Contents of the file
    # {
    #     "emo_adoration_sentences": "You're just the sweetest person I know and I am so happy to call you my friend. I had the best time with you, I just adore you. I love this gift, thank you!",
    #     "emo_amazement_sentences": "I just love how you can play guitar. You're so impressive. I admire your abilities so much.",
    #     ...
    # }
    utt2transcript = json.loads((corpus_dir / "transcripts.json").read_text())
    supervisions = []
    recordings_list = []
    for part in tqdm(dataset_parts, desc="Preparing EARS speakers"):
        if manifests_exist(part=part, output_dir=output_dir, prefix="ears"):
            logging.info(f"EARS subset: {part} already prepared - skipping.")
            continue
        spk_id = part
        part_path = corpus_dir / part
        recordings = RecordingSet.from_dir(
            part_path,
            "*.wav",
            num_jobs=num_jobs,
            recording_id=lambda path: f"{spk_id}_{path.stem}",
        )
        recordings_list.append(recordings)
        for rec in recordings:
            utt = rec.id.split("_")[1]
            meta = spk2meta[spk_id].copy()
            supervisions.append(
                SupervisionSegment(
                    id=rec.id,
                    recording_id=rec.id,
                    start=0.0,
                    duration=rec.duration,
                    channel=0,
                    text=utt2transcript.get(utt),
                    language="English",
                    speaker=spk_id,
                    gender=meta.pop("gender", None),
                    custom=meta,
                )
            )

    recordings = []
    for recs in recordings_list:
        recordings += list(recs)
    recordings = RecordingSet.from_recordings(recordings)
    supervisions = SupervisionSet.from_segments(supervisions)
    recordings, supervisions = fix_manifests(recordings, supervisions)
    validate_recordings_and_supervisions(recordings, supervisions)

    if output_dir is not None:
        supervisions.to_file(output_dir / f"ears_supervisions.jsonl.gz")
        recordings.to_file(output_dir / f"ears_recordings.jsonl.gz")

    manifests = {"recordings": recordings, "supervisions": supervisions}

    return manifests
