"""
About the medical corpus

A dataset of simulated patient-physician medical interviews with a focus on respiratory cases.
The simulated medical conversation dataset is available on figshare.com.
The dataset is divided into two sets of files: audio files of the simulated conversations in mp3 format, and the transcripts of the audio files as text files.
There are 272 mp3 audio files and 272 corresponding transcript text files.
Each file is titled with three characters and four digits.
RES stands for respiratory, GAS represents gastrointestinal, CAR is cardiovascular, MSK is musculoskeletal, DER is dermatological, and the four following digits represent the case number of the respective disease category.

It is covered in more detail at https://www.nature.com/articles/s41597-022-01423-1.pdf
"""

import logging
import os
import shutil
import tarfile
from collections import defaultdict
from concurrent.futures.process import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from tqdm.auto import tqdm

from lhotse.audio import Recording, RecordingSet
from lhotse.qa import fix_manifests, validate_recordings_and_supervisions
from lhotse.recipes.utils import manifests_exist
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, resumable_download

MEDICAL = ("test", "dev", "train")
MEDICAL_SPLITS = (
    "audio.tar.gz",
    "cleantext.tar.gz",
    "medical_test.info",
    "medical_dev.info",
    "medical_train.info",
)
MEDICAL_BASE_URL = "https://huggingface.co/datasets/yfyeung/medical/resolve/main/"


def download_medical(
    target_dir: Pathlike = ".",
    force_download: bool = False,
) -> Path:
    """
    Download and unzip Medical dataset.

    :param target_dir: Pathlike, the path of the dir to store the dataset.
    :param force_download: bool, if True, download the archive even if it already exists.

    :return: the path to downloaded and extracted directory with data.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    dataset_parts = MEDICAL_SPLITS

    for part in tqdm(dataset_parts, desc="Downloading Medical"):
        logging.info(f"Downloading part: {part}")
        # Process the archive.
        part_path = target_dir / part
        part_dir = str(part_path).replace(".tar.gz", "")
        resumable_download(
            MEDICAL_BASE_URL + part,
            filename=part_path,
            force_download=force_download,
        )
        # Remove partial unpacked files, if any, and unpack everything.
        if "tar.gz" in part:
            shutil.rmtree(part_dir, ignore_errors=True)
            with tarfile.open(part_path) as tar:
                tar.extractall(target_dir)

    return target_dir


def _parse_utterance(
    corpus_dir: Pathlike,
    audio_info: str,
) -> Optional[Tuple[Recording, SupervisionSegment]]:
    audio_path, start, end, text = (
        audio_info.replace(",", "\t").replace("[", "\t").replace("]", "").split("\t")
    )
    file_name = str(audio_path).replace(".mp3", "").replace("audio/", "")
    audio_path = (corpus_dir / audio_path).resolve()

    if not audio_path.is_file():
        logging.warning(f"No such file: {audio_path}")
        return None

    recording = Recording.from_file(path=audio_path, recording_id=file_name)
    segment = SupervisionSegment(
        id=file_name + "_" + str(hash(audio_info)),
        recording_id=file_name,
        start=float(start),
        duration=float(end) - float(start),
        channel=0,
        language="English",
        text=text,
    )

    return recording, segment


def _prepare_subset(
    subset: str,
    corpus_dir: Pathlike,
    num_jobs: int = 1,
) -> Tuple[RecordingSet, SupervisionSet]:
    """
    Returns the RecodingSet and SupervisionSet given a dataset part.
    :param subset: str, the name of the subset.
    :param corpus_dir: Pathlike, the path of the data dir.
    :return: the RecodingSet and SupervisionSet for train and valid.
    """
    corpus_dir = Path(corpus_dir)
    text_path = corpus_dir / ("medical_" + subset + ".info")

    with open(text_path) as f:
        audio_infos = f.read().splitlines()

    with ProcessPoolExecutor(num_jobs) as ex:
        futures = []
        recordings = []
        supervisions = []
        for audio_info in tqdm(audio_infos, desc="Distributing tasks"):
            futures.append(ex.submit(_parse_utterance, corpus_dir, audio_info))

        for future in tqdm(futures, desc="Processing"):
            result = future.result()
            if result is None:
                continue
            recording, segment = result
            if recording not in recordings:
                recordings.append(recording)
            supervisions.append(segment)

        recording_set = RecordingSet.from_recordings(recordings)
        supervision_set = SupervisionSet.from_segments(supervisions)

        # Fix manifests
        recording_set, supervision_set = fix_manifests(recording_set, supervision_set)
        validate_recordings_and_supervisions(recording_set, supervision_set)

    return recording_set, supervision_set


def prepare_medical(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions.
    :param corpus_dir: Path to the Medical dataset.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'recordings' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)

    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    logging.info("Preparing Medical...")

    subsets = MEDICAL

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    manifests = defaultdict(dict)

    for part in tqdm(subsets, desc="Dataset parts"):
        logging.info(f"Processing Medical subset: {part}")
        if manifests_exist(
            part=part,
            output_dir=output_dir,
            prefix="medical",
            suffix="jsonl.gz",
        ):
            logging.info(f"Medical subset: {part} already prepared - skipping.")
            continue

        recording_set, supervision_set = _prepare_subset(part, corpus_dir, num_jobs)

        if output_dir is not None:
            supervision_set.to_file(
                output_dir / f"medical_supervisions_{part}.jsonl.gz"
            )
            recording_set.to_file(output_dir / f"medical_recordings_{part}.jsonl.gz")

        manifests[part] = {"recordings": recording_set, "supervisions": supervision_set}

    return manifests
