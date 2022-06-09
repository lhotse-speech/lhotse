#!/usr/bin/env python3
#
# Copyright 2022 Xiaomi Corporation (Author: Mingshuang Luo)
# Apache 2.0
"""
TAL-CSASR is a multilingual ASR dataset which contains Mandarin and English
two languages. It is published by the TAL corporation. You can have a look
at this dataset by https://ai.100tal.com/dataset. Before using this recipe,
you have to download the dataset by yourself.
"""

import glob
import logging
import os
import zipfile
from collections import defaultdict
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional, Union

from tqdm import tqdm

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, urlretrieve_progress


def prepare_tal_csasr(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consists of the Recodings and Supervisions.
    :param corpus_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write and save the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'audio' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    manifests = defaultdict(dict)
    dataset_parts = ["train_set", "dev_set", "test_set"]

    # for part in tqdm(dataset_parts, desc="Dataset part"):
    for part in dataset_parts:
        logging.info(f"Prepare manifest for {part}")
        transcript_dict = {}
        with open(corpus_dir / f"{part}/label.txt", "r", encoding="utf-8") as f:
            for line in f:
                idx_transcript = line.split()
                transcript_dict[idx_transcript[0]] = " ".join(idx_transcript[1:])

        recordings = []
        supervisions = []
        wav_path = corpus_dir / f"{part}" / "wav"
        wav_files = glob.glob(os.path.join(wav_path, "*.wav"))
        for i in tqdm(range(len(wav_files))):
            audio_path = wav_files[i]
            idx = Path(audio_path).stem
            if idx not in transcript_dict:
                logging.warning(f"No transcript: {idx}")
                continue
            text = transcript_dict[idx]
            if not Path(audio_path).is_file():
                logging.warning(f"No such file: {audio_path}")
                continue
            recording = Recording.from_file(path=audio_path, recording_id=idx)
            recordings.append(recording)
            segment = SupervisionSegment(
                id=idx,
                recording_id=idx,
                start=0.0,
                duration=recording.duration,
                channel=0,
                language="English-Chinese",
                text=text.strip(),
            )

            supervisions.append(segment)

        recording_set = RecordingSet.from_recordings(recordings)
        supervision_set = SupervisionSet.from_segments(supervisions)
        validate_recordings_and_supervisions(recording_set, supervision_set)

        if output_dir is not None:
            supervision_set.to_file(
                output_dir / f"tal_csasr_supervisions_{part}.jsonl.gz"
            )
            recording_set.to_file(output_dir / f"tal_csasr_recordings_{part}.jsonl.gz")

        manifests[part] = {
            "recordings": recording_set,
            "supervisions": supervision_set,
        }

    return manifests
