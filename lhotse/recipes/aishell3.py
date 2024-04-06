"""
AISHELL-3 is a large-scale and high-fidelity multi-speaker Mandarin speech corpus
published by Beijing Shell Shell Technology Co.,Ltd.
It can be used to train multi-speaker Text-to-Speech (TTS) systems.
The corpus contains roughly 85 hours of emotion-neutral recordings spoken by
218 native Chinese mandarin speakers and total 88035 utterances.
Their auxiliary attributes such as gender, age group and native accents are
explicitly marked and provided in the corpus. Accordingly, transcripts in Chinese
character-level and pinyin-level are provided along with the recordings.
The word & tone transcription accuracy rate is above 98%, through professional
speech annotation and strict quality inspection for tone and prosody.
"""
import logging
import shutil
import tarfile
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

from tqdm import tqdm

from lhotse import (
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
    validate_recordings_and_supervisions,
)
from lhotse.audio import Recording, RecordingSet
from lhotse.qa import fix_manifests
from lhotse.recipes.utils import manifests_exist, read_manifests_if_cached
from lhotse.utils import Pathlike, resumable_download, safe_extract

aishell3 = (
    "test",
    "train",
)


def download_aishell3(
    target_dir: Pathlike = ".",
    force_download: Optional[bool] = False,
    base_url: Optional[str] = "http://www.openslr.org/resources",
) -> Path:
    """
    Download and untar the dataset

    :param target_dir: Pathlike, the path of the dir to storage the dataset.
    :param force_download: Bool, if True, download the tars no matter if the tars exist.
    :param base_url: str, the url of the OpenSLR resources.
    :return: the path to downloaded and extracted directory with data.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    url = f"{base_url}/93"
    tar_name = "data_aishell3.tgz"
    tar_path = target_dir / tar_name
    completed_detector = target_dir / ".completed"
    if completed_detector.is_file():
        logging.info(f"Skipping {tar_name} because {completed_detector} exists.")
        return target_dir
    resumable_download(
        f"{url}/{tar_name}", filename=tar_path, force_download=force_download
    )
    with tarfile.open(tar_path) as tar:
        safe_extract(tar, path=target_dir)
    completed_detector.touch()
    return target_dir


def prepare_aishell3(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions.
    When all the manifests are available in the ``output_dir``, it will simply read and return them.

    :param corpus_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'audio' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    dataset_parts = aishell3
    manifests = {}

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Maybe the manifests already exist: we can read them and save a bit of preparation time.
        manifests = read_manifests_if_cached(
            dataset_parts=dataset_parts, output_dir=output_dir, prefix="aishell3"
        )

    speaker_info = {}
    speaker_info_path = corpus_dir / "spk-info.txt"
    assert speaker_info_path.is_file(), f"No such file: {speaker_info_path}"
    with open(speaker_info_path, "r") as f:
        for k in f.readlines():
            k = k.strip()
            if k.startswith("#") or len(k) == 0:
                continue
            k = k.split("\t")
            speaker, gender = k[0], k[2]
            speaker_info[speaker] = gender

    label_path = corpus_dir / "train" / "label_train-set.txt"
    assert label_path.is_file(), f"No such file: {label_path}"
    with open(label_path, "r") as f:
        tone_labels = {}
        for k in f.readlines():
            k = k.strip()
            if k.startswith("#") or len(k) == 0:
                continue
            k = k.split("|")
            assert len(k) == 3
            tone_labels[k[0]] = k[1:]

    for part in tqdm(dataset_parts, desc="Preparing aishell3 parts"):
        if manifests_exist(part=part, output_dir=output_dir, prefix="aishell3"):
            logging.info(f"aishell3 subset: {part} already prepared - skipping.")
            continue
        part_path = corpus_dir / part
        scripts_path = part_path / "content.txt"
        assert scripts_path.is_file(), f"No such file: {scripts_path}"
        recordings = []
        supervisions = []
        with open(scripts_path) as f:
            for line in tqdm(f):
                id, text = line.strip().split("\t")
                audio_path = part_path / "wav" / id[:7] / id
                id = id.split(".")[0]
                tones = tone_labels.get(id, (None, None))
                speaker = id[:7]
                text = "".join([x for i, x in enumerate(text.split()) if i % 2 == 0])
                pinyin = " ".join([x for i, x in enumerate(text.split()) if i % 2 == 1])
                if not audio_path.is_file():
                    logging.warning(f"No such file: {audio_path}")
                    continue
                recording = Recording.from_file(audio_path)
                segment = SupervisionSegment(
                    id=id,
                    recording_id=id,
                    start=0.0,
                    duration=recording.duration,
                    channel=0,
                    language="Chinese",
                    speaker=speaker,
                    gender=speaker_info.get(speaker, "female"),
                    text=text,
                    custom={
                        "pinyin": pinyin.strip(),
                        "tones_pinyin": tones[0],
                        "tones_text": tones[1],
                    },
                )
                recordings.append(recording)
                supervisions.append(segment)

        recording_set = RecordingSet.from_recordings(recordings)
        supervision_set = SupervisionSet.from_segments(supervisions)
        recording_set, supervision_set = fix_manifests(recording_set, supervision_set)
        validate_recordings_and_supervisions(recording_set, supervision_set)

        if output_dir is not None:
            supervision_set.to_file(
                output_dir / f"aishell3_supervisions_{part}.jsonl.gz"
            )
            recording_set.to_file(output_dir / f"aishell3_recordings_{part}.jsonl.gz")

        manifests[part] = {"recordings": supervision_set, "supervisions": recording_set}

    return manifests
