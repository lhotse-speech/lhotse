"""
Magicdata is an open-source Chinese Mandarin speech corpus by Magic Data Technology Co., Ltd.,
containing 755 hours of scripted read speech data from 1080 native speakers of the Mandarin Chinese spoken
in mainland China. The sentence transcription accuracy is higher than 98%.
Publicly available on https://www.openslr.org/resources/68
MagicData (712 hours)
"""


import logging
import os
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Union

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, urlretrieve_progress


def download_magicdata(
    target_dir: Pathlike = ".",
    force_download: Optional[bool] = False,
    base_url: Optional[str] = "http://www.openslr.org/resources",
) -> Path:
    """
    Downdload and untar the dataset
    :param target_dir: Pathlike, the path of the dir to storage the dataset.
    :param force_download: Bool, if True, download the tars no matter if the tars exist.
    :param base_url: str, the url of the OpenSLR resources.
    :return: the path to downloaded and extracted directory with data.
    """
    url = f"{base_url}/68"
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir = target_dir / "magicdata"
    train_tar_name = "train_set.tar.gz"
    dev_tar_name = "dev_set.tar.gz"
    test_tar_name = "test_set.tar.gz"
    for tar_name in [train_tar_name, dev_tar_name, test_tar_name]:
        tar_path = target_dir / tar_name
        extracted_dir = corpus_dir / tar_name[:-7]
        completed_detector = extracted_dir / ".completed"
        if completed_detector.is_file():
            logging.info(f"Skipping download of because {completed_detector} exists.")
            continue
        if force_download or not tar_path.is_file():
            urlretrieve_progress(
                f"{url}/{tar_name}", filename=tar_path, desc=f"Downloading {tar_name}"
            )
        shutil.rmtree(extracted_dir, ignore_errors=True)
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=corpus_dir)
        completed_detector.touch()

    return corpus_dir


def prepare_magicdata(
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

    train_transcript_path = corpus_dir / "train/TRANS.txt"
    train_transcript_dict = {}
    with open(train_transcript_path, "r", encoding="utf-8") as f1:
        for line in f1.readlines():
            idx_transcript = line.split()
            train_transcript_dict[idx_transcript[0]] = " ".join(idx_transcript[1:])

    dev_transcript_path = corpus_dir / "dev/TRANS.txt"
    dev_transcript_dict = {}
    with open(dev_transcript_path, "r", encoding="utf-8") as f2:
        for line in f2.readlines():
            idx_transcript = line.split()
            dev_transcript_dict[idx_transcript[0]] = " ".join(idx_transcript[1:])

    test_transcript_path = corpus_dir / "test/TRANS.txt"
    test_transcript_dict = {}
    with open(test_transcript_path, "r", encoding="utf-8") as f3:
        for line in f3.readlines():
            idx_transcript = line.split()
            test_transcript_dict[idx_transcript[0]] = " ".join(idx_transcript[1:])

    manifests = defaultdict(dict)
    dataset_parts = ["train", "dev", "test"]
    for part in dataset_parts:
        # Generate a mapping: utt_id -> (audio_path, audio_info, speaker, text)
        recordings = []
        supervisions = []
        wav_path = corpus_dir / f"{part}"
        for audio_path in wav_path.rglob("**/*.wav"):
            idx = audio_path.stem
            speaker = audio_path.parts[-2]
            specify_transcript = f"{part}_transcript_dict"
            if idx not in specify_transcript:
                logging.warning(f"No transcript: {idx}")
                continue

            text = specify_transcript[idx]
            if not audio_path.is_file():
                logging.warning(f"No such file: {audio_path}")
                continue
            recording = Recording.from_file(audio_path)
            recordings.append(recording)
            segment = SupervisionSegment(
                id=idx,
                recording_id=idx,
                start=0.0,
                duration=recording.duration,
                channel=0,
                language="Chinese",
                speaker=speaker,
                text=text.strip(),
            )
            supervisions.append(segment)

        recording_set = RecordingSet.from_recordings(recordings)
        supervision_set = SupervisionSet.from_segments(supervisions)
        validate_recordings_and_supervisions(recording_set, supervision_set)

        if output_dir is not None:
            supervision_set.to_file(
                output_dir / f"magicdata_supervisions_{part}.jsonl.gz"
            )
            recording_set.to_file(output_dir / f"magicdata_recordings_{part}.jsonl.gz")

        manifests[part] = {"recordings": recording_set, "supervisions": supervision_set}

    return manifests
