"""
Stcmds is an open-source  Chinese Mandarin corpus by Surfingtech (www.surfing.ai), containing utterances from 855 speakers, 102600 utterances;
Publicly available on https://www.openslr.org/resources/38
ST-CMDS (110 hours)

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
    """
    Modified from https://github.com/wenet-e2e/wenet/blob/main/examples/multi_cn/s0/local/stcmds_data_prep.sh#L42
    paste -d' ' $data/utt.list $data/text.list |\
    sed 's/，//g' |\
    tr '[a-z]' '[A-Z]' |\
    awk '{if (NF > 1) print $0;}' > $data/train/text
    """
    line = line.replace("，", "")
    line = line.upper()
    return line


def download_stcmds(
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
    url = f"{base_url}/38"
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir = target_dir / "stcmds"
    dataset_tar_name = "ST-CMDS-20170001_1-OS.tar.gz"
    for tar_name in [dataset_tar_name]:
        tar_path = target_dir / tar_name
        extracted_dir = corpus_dir / tar_name[:-7]
        completed_detector = extracted_dir / ".completed"
        if completed_detector.is_file():
            logging.info(f"Skipping download of because {completed_detector} exists.")
            continue
        resumable_download(
            f"{url}/{tar_name}", filename=tar_path, force_download=force_download
        )
        shutil.rmtree(extracted_dir, ignore_errors=True)
        with tarfile.open(tar_path) as tar:
            safe_extract(tar, path=corpus_dir)
        completed_detector.touch()

    return corpus_dir


def prepare_stcmds(
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

    path = corpus_dir / "ST-CMDS-20170001_1-OS"
    transcript_dict = {}
    for text_path in path.rglob("**/*.txt"):
        idx = text_path.stem
        logging.info(f"processing stcmds transcript  {text_path}")
        with open(text_path, "r", encoding="utf-8") as f:
            for line in f:
                line = text_normalize(line)
                transcript_dict[idx] = line

    manifests = defaultdict(dict)
    dataset_parts = ["train"]
    for part in tqdm(
        dataset_parts,
        desc="process stcmds audio, it needs waste about 2169 seconds time.",
    ):
        logging.info(f"Processing stcmds {part}")
        recordings = []
        supervisions = []
        for audio_path in path.rglob("**/*.wav"):
            idx = audio_path.stem
            speaker = "".join(list(idx)[8:15])
            if idx not in transcript_dict:
                logging.warning(f"No transcript: {idx}")
                logging.warning(f"{audio_path} has no transcript")
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
            supervision_set.to_file(output_dir / f"stcmds_supervisions_{part}.jsonl.gz")
            recording_set.to_file(output_dir / f"stcmds_recordings_{part}.jsonl.gz")

        manifests[part] = {"recordings": recording_set, "supervisions": supervision_set}

    return manifests
