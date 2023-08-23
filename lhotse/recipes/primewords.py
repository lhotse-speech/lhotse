"""
Primewords is an open-source Chinese Mandarin corpus released by Shanghai Primewords Co. Ltd.
Publicly available on https://www.openslr.org/47/
Primewords (99 hours)
"""
import json
import logging
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


def download_primewords(
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
    url = f"{base_url}/47"
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir = target_dir / "primewords"
    dataset_tar_name = "primewords_md_2018_set1.tar.gz"
    for tar_name in [dataset_tar_name]:
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


def prepare_primewords(
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
    transcript_path = corpus_dir / "primewords_md_2018_set1" / "set1_transcript.json"
    transcript_dict = {}
    speaker_dict = {}
    with open(transcript_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for utt in data:
            content = utt["text"]
            uttid = utt["file"].split(".")[0]
            user_id = utt["user_id"]
            transcript_dict[uttid] = content
            speaker_dict[uttid] = user_id

    manifests = defaultdict(dict)
    dataset_parts = ["train"]
    for part in tqdm(
        dataset_parts, desc="Process primewords audio, it takes 35 seconds."
    ):
        logging.info(f"Processing primewords  subset: {part}")
        # Generate a mapping: utt_id -> (audio_path, audio_info, speaker, text)
        recordings = []
        supervisions = []
        wav_path = corpus_dir / "primewords_md_2018_set1" / "audio_files"
        for audio_path in wav_path.rglob("**/*.wav"):

            idx = audio_path.stem
            speaker = speaker_dict[idx]
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
                output_dir / f"primewords_supervisions_{part}.jsonl.gz"
            )
            recording_set.to_file(output_dir / f"primewords_recordings_{part}.jsonl.gz")

        manifests[part] = {"recordings": recording_set, "supervisions": supervision_set}

    return manifests
