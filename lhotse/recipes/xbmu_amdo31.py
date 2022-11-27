"""
About the XBMU-AMDO31 corpus
XBMU-AMDO31 is an open-source Amdo Tibetan speech corpus published by Northwest Minzu University.
publicly available on https://huggingface.co/datasets/syzym/xbmu_amdo31
"""

import logging
import os
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Union

from tqdm.auto import tqdm

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, safe_extract, urlretrieve_progress


def download_xbmu_amdo31(
    target_dir: Pathlike = ".",
) -> Path:
    """
    Downdload and untar the dataset
    :param target_dir: Pathlike, the path of the dir to storage the dataset.
    :param force_download: Bool, if True, download the tars no matter if the tars exist.
    :param base_url: str, the url of the OpenSLR resources.
    :return: the path to downloaded and extracted directory with data.
    """
    target_dir = Path(target_dir)
    corpus_dir = target_dir / "xbmu_amdo31"
    wav_dir = corpus_dir / "data" / "wav"
    train_tar_name = "train.tar.gz"
    dev_tar_name = "dev.tar.gz"
    test_tar_name = "test.tar.gz"

    for tar_name in [train_tar_name, dev_tar_name, test_tar_name]:
        tar_path = wav_dir / tar_name
        extracted_dir = wav_dir / tar_name[:-7]
        completed_detector = extracted_dir / ".completed"
        if completed_detector.is_file():
            logging.info(f"Skipping tar of because {completed_detector} exists.")
            continue

        shutil.rmtree(extracted_dir, ignore_errors=True)
        with tarfile.open(tar_path) as tar:
            safe_extract(tar, path=wav_dir)
        completed_detector.touch()

    return corpus_dir

def prepare_xbmu_amdo31(
    corpus_dir: Pathlike, output_dir: Optional[Pathlike] = None
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'recordings' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = corpus_dir / "data/transcript/transcript_clean.txt"
    transcript_dict = {}
    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            idx_transcript = line.split()
            content = " ".join(idx_transcript[1:])
            transcript_dict[idx_transcript[0]] = content
    manifests = defaultdict(dict)
    dataset_parts = ["train", "dev", "test"]
    for part in tqdm(
        dataset_parts,
        desc="Process xbmu_amdo31 audio.",
    ):
        logging.info(f"Processing xbmu_amdo31 subset: {part}")
        # Generate a mapping: utt_id -> (audio_path, audio_info, speaker, text)
        recordings = []
        supervisions = []
        wav_path = corpus_dir / "data" / "wav" / f"{part}"
        count = 0
        for audio_path in wav_path.rglob("**/*.wav"):
            idx = audio_path.stem.split("-")[1]
            speaker = audio_path.parts[-2]
            if idx not in transcript_dict:
                logging.warning(f"No transcript: {idx}")
                logging.warning(f"{audio_path} has no transcript.")
                continue
            text = transcript_dict[idx]
            if not audio_path.is_file():
                logging.warning(f"No such file: {audio_path}")
                continue
            recording = Recording.from_file(audio_path)
            recordings.append(recording)
            count += 1
            segment = SupervisionSegment(
                id=str(count) +  "_" + idx,
                recording_id=speaker + "-" +idx,
                start=0.0,
                duration=recording.duration,
                channel=0,
                language="tibetan",
                speaker=speaker,
                text=text.strip(),
            )
            supervisions.append(segment)

        recording_set = RecordingSet.from_recordings(recordings)
        supervision_set = SupervisionSet.from_segments(supervisions)
        validate_recordings_and_supervisions(recording_set, supervision_set)

        if output_dir is not None:
            supervision_set.to_file(
                output_dir / f"xbmu_amdo31_supervisions_{part}.jsonl.gz"
            )
            recording_set.to_file(output_dir / f"xbmu_amdo31_recordings_{part}.jsonl.gz")

        manifests[part] = {"recordings": recording_set, "supervisions": supervision_set}

    return manifests