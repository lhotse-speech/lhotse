"""
Magicdata is an open-source Chinese Mandarin speech corpus by Magic Data Technology Co., Ltd.,
containing 755 hours of scripted read speech data from 1080 native speakers of the Mandarin Chinese spoken
in mainland China. The sentence transcription accuracy is higher than 98%.
Publicly available on https://www.openslr.org/68
"""


import logging
import os
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Union

from tqdm.auto import tqdm

from lhotse import fix_manifests, validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, resumable_download, safe_extract


def text_normalize(line: str):
    r"""
    Modified from https://github.com/wenet-e2e/wenet/blob/main/examples/multi_cn/s0/local/magicdata_data_prep.sh#L41
    sed 's/！//g' | sed 's/？//g' |\
    sed 's/，//g' | sed 's/－//g' |\
    sed 's/：//g' | sed 's/；//g' |\
    sed 's/　//g' | sed 's/。//g' |\
    sed 's/`//g' | sed 's/,//g' |\
    sed 's/://g' | sed 's/?//g' |\
    sed 's/\///g' | sed 's/·//g' |\
    sed 's/\"//g' | sed 's/“//g' |\
    sed 's/”//g' | sed 's/\\//g' |\
    sed 's/…//g' | sed "s///g" |\
    sed 's/、//g' | sed "s///g" | sed 's/《//g' | sed 's/》//g' |\
    sed 's/\[//g' | sed 's/\]//g' | sed 's/FIL//g' | sed 's/SPK//' |\
    tr '[a-z]' '[A-Z]' |\
    """
    line = line.replace("！", "")
    line = line.replace("？", "")
    line = line.replace("，", "")
    line = line.replace("－", "")
    line = line.replace("：", "")
    line = line.replace("；", "")
    line = line.replace("　", "")
    line = line.replace("。", "")
    line = line.replace("`", "")
    line = line.replace(",", "")
    line = line.replace(":", "")
    line = line.replace("?", "")
    line = line.replace("/", "")
    line = line.replace("·", "")
    line = line.replace('"', "")
    line = line.replace("“", "")
    line = line.replace("”", "")
    line = line.replace("\\", "")
    line = line.replace("…", "")
    line = line.replace("、", "")
    line = line.replace("[ ", "")
    line = line.replace("[", "")
    line = line.replace("]", "")
    line = line.replace("《 ", "")
    line = line.replace("《", "")
    line = line.replace("》", "")
    line = line.replace("FIL", "")
    line = line.replace("SPK", "")
    line = line.replace("﻿", "")
    line = line.upper()
    return line


def download_magicdata(
    target_dir: Pathlike = ".",
    force_download: bool = False,
    base_url: str = "http://www.openslr.org/resources",
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
            logging.info(
                f"Skipping download {tar_name} because {completed_detector} exists."
            )
            continue
        resumable_download(
            f"{url}/{tar_name}", filename=tar_path, force_download=force_download
        )
        shutil.rmtree(extracted_dir, ignore_errors=True)
        with tarfile.open(tar_path) as tar:
            safe_extract(tar, path=corpus_dir)
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

    dataset_parts = ["train", "dev", "test"]
    transcript_dict = {}
    for part in dataset_parts:
        path = corpus_dir / f"{part}" / "TRANS.txt"
        with open(path, "r", encoding="utf-8") as f1:
            for line in f1:
                if line.startswith("UtteranceID"):
                    logging.info(f"line is {line}")
                    continue
                idx_transcript = line.split()
                ## because the below two utterances are bad, they are removed.
                if (
                    idx_transcript[0] == "16_4013_20170819121429.wav"
                    or idx_transcript[0] == "18_1565_20170712000170.wav"
                ):
                    continue
                idx_ = idx_transcript[0].split(".")[0]
                content = " ".join(idx_transcript[2:])
                content = text_normalize(content)
                transcript_dict[idx_] = content

    manifests = defaultdict(dict)
    for part in tqdm(
        dataset_parts, desc="Process magicdata audio, it takes 6818 seconds."
    ):
        logging.info(f"Processing magicdata subset: {part}")
        # Generate a mapping: utt_id -> (audio_path, audio_info, speaker, text)
        recordings = []
        supervisions = []
        wav_path = corpus_dir / f"{part}"
        for audio_path in wav_path.rglob("**/*.wav"):
            idx = audio_path.stem
            speaker = audio_path.parts[-2]

            if idx not in transcript_dict:
                logging.warning(f"No transcript: {idx}")
                logging.info(f"{audio_path} has no transcript.")
                continue

            text = transcript_dict[idx]
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

        recording_set, supervision_set = fix_manifests(recording_set, supervision_set)
        validate_recordings_and_supervisions(recording_set, supervision_set)

        if output_dir is not None:
            supervision_set.to_file(
                output_dir / f"magicdata_supervisions_{part}.jsonl.gz"
            )
            recording_set.to_file(output_dir / f"magicdata_recordings_{part}.jsonl.gz")

        manifests[part] = {"recordings": recording_set, "supervisions": supervision_set}

    return manifests
