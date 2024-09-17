"""
The SpeechIO Chinese data is a collection of test sets covering wide range of speech recognition tasks & scenarios.

Participants can obtain the datasets at https://github.com/SpeechColab/Leaderboard - please download the datasets manually.
"""

import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from tqdm.auto import tqdm

from lhotse.audio import AudioSource, Recording, RecordingSet
from lhotse.qa import fix_manifests, validate_recordings_and_supervisions
from lhotse.recipes.utils import manifests_exist
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, is_module_available

SPEECHIO_TESTSET_INDEX = 26  # Currently, from 0 - 26 test sets are open source.


def _parse_one_subset(
    corpus_dir: Pathlike,
) -> Optional[Tuple[Recording, SupervisionSegment]]:
    recordings = []
    segments = []

    if not is_module_available("pandas"):
        raise ValueError("To prepare speechio data, please 'pip install pandas' first.")
    import pandas as pd

    df = pd.read_csv(f"{str(corpus_dir)}/metadata.tsv", sep="\t")

    recording_ids = df["ID"].tolist()
    texts = df["TEXT"].tolist()
    wav_paths = df["AUDIO"].tolist()

    for idx, audio_path in enumerate(wav_paths):
        audio_path = str(corpus_dir / audio_path)
        if not os.path.exists(audio_path):
            logging.warning(f"Audio file {audio_path} does not exist - skipping.")
            continue
        recording = Recording.from_file(audio_path)
        recordings.append(recording)
        recording_id = recording_ids[idx]
        text = texts[idx]
        speaker = recording_id.split("_")[0]

        segment = SupervisionSegment(
            id=f"{corpus_dir}-{recording_id}",
            recording_id=recording_id,
            start=0,
            duration=recording.duration,
            channel=0,
            language="Chinese",
            speaker=speaker,
            text=text,
        )
        segments.append(segment)

    return recordings, segments


def _prepare_subset(
    subset: str,
    corpus_dir: Pathlike,
) -> Tuple[RecordingSet, SupervisionSet]:
    """
    Returns the RecodingSet and SupervisionSet given a dataset part.
    :param subset: str, the name of the subset.
    :param corpus_dir: Pathlike, the path of the data dir.
    :return: the RecodingSet and SupervisionSet for train and valid.
    """
    corpus_dir = Path(corpus_dir)
    part_path = corpus_dir / subset

    recording_set, supervision_set = _parse_one_subset(part_path)
    recording_set = RecordingSet.from_recordings(recording_set)
    supervision_set = SupervisionSet.from_segments(supervision_set)

    # Fix manifests
    recording_set, supervision_set = fix_manifests(recording_set, supervision_set)
    validate_recordings_and_supervisions(recording_set, supervision_set)

    return recording_set, supervision_set


def prepare_speechio(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Path to the SpeechIO dataset.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'recordings' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)

    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    logging.info("Preparing SpeechIO...")

    subsets = []
    for i in range(SPEECHIO_TESTSET_INDEX + 1):
        idx = f"{i}".zfill(2)
        subsets.append(f"SPEECHIO_ASR_ZH000{idx}")

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    manifests = defaultdict(dict)

    for part in tqdm(subsets, desc="Dataset parts"):
        logging.info(f"Processing SpeechIO subset: {part}")
        if manifests_exist(
            part=part,
            output_dir=output_dir,
            prefix=f"speechio",
            suffix="jsonl.gz",
        ):
            logging.info(f"SpeechIO subset: {part} already prepared - skipping.")
            continue

        recording_set, supervision_set = _prepare_subset(part, corpus_dir)

        if output_dir is not None:
            supervision_set.to_file(
                output_dir / f"speechio_supervisions_{part}.jsonl.gz"
            )
            recording_set.to_file(output_dir / f"speechio_recordings_{part}.jsonl.gz")

        manifests[part] = {"recordings": recording_set, "supervisions": supervision_set}

    return manifests
