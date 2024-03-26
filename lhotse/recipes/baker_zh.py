"""
See https://en.data-baker.com/datasets/freeDatasets/

It is a Chinese TTS dataset, containing 12 hours of data.
"""

import logging
import re
import shutil
import tarfile
from pathlib import Path
from typing import Dict, Optional, Union

from lhotse import fix_manifests, validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, resumable_download, safe_extract


def download_baker_zh(
    target_dir: Pathlike = ".", force_download: Optional[bool] = False
) -> Path:
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = "BZNSYP"
    tar_path = target_dir / f"{dataset_name}.tar.bz2"
    corpus_dir = target_dir / dataset_name
    completed_detector = corpus_dir / ".completed"
    if completed_detector.is_file():
        logging.info(f"Skipping {dataset_name} because {completed_detector} exists.")
        return corpus_dir
    resumable_download(
        f"https://huggingface.co/openspeech/BZNSYP/resolve/main/{dataset_name}.tar.bz2",
        filename=tar_path,
        force_download=force_download,
    )
    shutil.rmtree(corpus_dir, ignore_errors=True)
    with tarfile.open(tar_path) as tar:
        safe_extract(tar, path=target_dir)
    completed_detector.touch()

    return corpus_dir


def prepare_baker_zh(
    corpus_dir: Pathlike, output_dir: Optional[Pathlike] = None
) -> Dict[str, Union[RecordingSet, SupervisionSet]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions

    :param corpus_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: The RecordingSet and SupervisionSet with the keys 'audio' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # The corpus_dir contains three sub directories
    # PhoneLabeling  ProsodyLabeling  Wave

    # Generate a mapping: utt_id -> (audio_path, audio_info, text)
    labeling_file = corpus_dir / "ProsodyLabeling" / "000001-010000.txt"
    if not labeling_file.is_file():
        raise ValueError(f"{labeling_file} does not exist")

    recordings = []
    supervisions = []
    logging.info("Started preparing. It may take 30 seconds")
    pattern = re.compile("#[12345]")
    with open(labeling_file) as f:
        try:
            while True:
                first = next(f).strip()
                pinyin = next(f).strip()
                recording_id, original_text = first.split(None, maxsplit=1)
                normalized_text = re.sub(pattern, "", original_text)
                audio_path = corpus_dir / "Wave" / f"{recording_id}.wav"

                if not audio_path.is_file():
                    logging.warning(f"No such file: {audio_path}")
                    continue
                recording = Recording.from_file(audio_path)

                segment = SupervisionSegment(
                    id=recording_id,
                    recording_id=recording_id,
                    start=0.0,
                    duration=recording.duration,
                    channel=0,
                    language="Chinese",
                    gender="female",
                    text=original_text,
                    custom={"pinyin": pinyin, "normalized_text": normalized_text},
                )
                recordings.append(recording)
                supervisions.append(segment)
        except StopIteration:
            pass

    recording_set = RecordingSet.from_recordings(recordings)
    supervision_set = SupervisionSet.from_segments(supervisions)

    recording_set, supervision_set = fix_manifests(recording_set, supervision_set)
    validate_recordings_and_supervisions(recording_set, supervision_set)

    if output_dir is not None:
        supervision_set.to_file(output_dir / "baker_zh_supervisions_all.jsonl.gz")
        recording_set.to_file(output_dir / "baker_zh_recordings_all.jsonl.gz")

    return {"recordings": recording_set, "supervisions": supervision_set}
