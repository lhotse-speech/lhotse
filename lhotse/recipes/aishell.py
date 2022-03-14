"""
About the Aishell corpus
Aishell is an open-source Chinese Mandarin speech corpus published by Beijing Shell Shell Technology Co.,Ltd.
publicly available on https://www.openslr.org/33
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


def download_aishell(
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
    url = f"{base_url}/33"
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir = target_dir / "aishell"
    dataset_tar_name = "data_aishell.tgz"
    resources_tar_name = "resource_aishell.tgz"
    for tar_name in [dataset_tar_name, resources_tar_name]:
        tar_path = target_dir / tar_name
        extracted_dir = corpus_dir / tar_name[:-4]
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
        if tar_name == dataset_tar_name:
            wav_dir = extracted_dir / "wav"
            for sub_tar_name in os.listdir(wav_dir):
                with tarfile.open(wav_dir / sub_tar_name) as tar:
                    tar.extractall(path=wav_dir)
        completed_detector.touch()

    return corpus_dir


def prepare_aishell(
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
    transcript_path = corpus_dir / "data_aishell/transcript/aishell_transcript_v0.8.txt"
    transcript_dict = {}
    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            idx_transcript = line.split()
            transcript_dict[idx_transcript[0]] = " ".join(idx_transcript[1:])
    manifests = defaultdict(dict)
    dataset_parts = ["train", "dev", "test"]
    for part in dataset_parts:
        # Generate a mapping: utt_id -> (audio_path, audio_info, speaker, text)
        recordings = []
        supervisions = []
        wav_path = corpus_dir / "data_aishell" / "wav" / f"{part}"
        for audio_path in wav_path.rglob("**/*.wav"):
            idx = audio_path.stem
            speaker = audio_path.parts[-2]
            if idx not in transcript_dict:
                logging.warning(f"No transcript: {idx}")
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
        validate_recordings_and_supervisions(recording_set, supervision_set)

        if output_dir is not None:
            supervision_set.to_json(output_dir / f"supervisions_{part}.json")
            recording_set.to_json(output_dir / f"recordings_{part}.json")

        manifests[part] = {"recordings": recording_set, "supervisions": supervision_set}

    return manifests
