"""
About the librilight corpus
Libri-light is a benchmark for the training of automatic speech recognition (ASR) systems with limited or no supervision.
It contains a large dataset of 60K hours of unlabelled speech from audiobooks in English and a small labelled dataset (10h, 1h, and 10 min) plus metrics, trainable baseline models, and pretrained models that use these datasets.
It is covered in more detail at https://arxiv.org/abs/1912.07875.
"""

import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from tqdm.auto import tqdm

from lhotse.audio import Recording, RecordingSet
from lhotse.recipes.utils import manifests_exist
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike

_SPLITS = ["small", "medium", "large"]


def _prepare_subset(
    subset: str,
    corpus_dir: Pathlike,
) -> Tuple[RecordingSet, SupervisionSet]:
    """
    Returns the RecodingSet and SupervisionSet given a dataset part.
    :param corpus_dir: Pathlike, the path of the data dir.
    :return: the RecodingSet and SupervisionSet for train and valid.
    """
    corpus_dir = Path(corpus_dir)
    part_path = corpus_dir / subset
    audio_paths = []
    for root, dirs, files in os.walk(part_path):
        if len(dirs) == 0:
            audio_paths += [
                os.path.join(root, file_path)
                for file_path in files
                if file_path.endswith(".flac")
            ]

    recordings = []
    supervisions = []
    for audio_path in audio_paths:
        file_name = audio_path.replace(".flac", "").replace(str(corpus_dir) + "/", "")
        speaker = audio_path.split("/")[-3]
        text = ""
        audio_path = Path(audio_path).resolve()

        if not audio_path.is_file():
            logging.warning(f"No such file: {audio_path}")
            continue

        recording = Recording.from_file(
            path=audio_path,
            recording_id=file_name,
        )
        recordings.append(recording)
        segment = SupervisionSegment(
            id=file_name,
            recording_id=file_name,
            start=0.0,
            duration=recording.duration,
            channel=0,
            language="English",
            speaker=speaker,
            text=text,
        )
        supervisions.append(segment)
    recording_set = RecordingSet.from_recordings(recordings)
    supervision_set = SupervisionSet.from_segments(supervisions)

    return recording_set, supervision_set


def prepare_librilight(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Path to the LibriLight dataset.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'recordings' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    output_dir = Path(output_dir) if output_dir is not None else None

    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    logging.info("Preparing LibriLight...")

    subsets = _SPLITS

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    manifests = defaultdict(dict)

    for part in tqdm(subsets, desc="Dataset parts"):
        logging.info(f"Processing LibriLight subset: {part}")
        if manifests_exist(
            part=part,
            output_dir=output_dir,
            prefix="librilight",
            suffix="jsonl.gz",
        ):
            logging.info(f"LibriLight subset: {part} already prepared - skipping.")
            continue

        recording_set, supervision_set = _prepare_subset(part, corpus_dir)

        if output_dir is not None:
            supervision_set.to_file(
                output_dir / f"librilight_supervisions_{part}.jsonl.gz"
            )
            recording_set.to_file(output_dir / f"librilight_recordings_{part}.jsonl.gz")

        manifests[part] = {"recordings": recording_set, "supervisions": supervision_set}

    return manifests
