"""
About the MobvoiHotwords corpus

    The MobvoiHotwords dataset is a ~144-hour corpus of wake word corpus which is
    publicly availble on https://www.openslr.org/87

    For wake word data, wake word utterances contain either 'Hi xiaowen' or 'Nihao
    Wenwen' are collected. For each wake word, there are about 36k utterances. All
    wake word data is collected from 788 subjects, ages 3-65, with different
    distances from the smart speaker (1, 3 and 5 meters). Different noises
    (typical home environment noises like music and TV) with varying sound
    pressure levels are played in the background during the collection.
"""

import json
import logging
import shutil
import tarfile
from pathlib import Path
from typing import Dict, Optional, Union

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.recipes.utils import manifests_exist, read_manifests_if_cached
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, urlretrieve_progress


def download_mobvoihotwords(
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

    url = f"{base_url}/87"
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir = target_dir / "MobvoiHotwords"
    dataset_tar_name = "mobvoi_hotword_dataset.tgz"
    resources_tar_name = "mobvoi_hotword_dataset_resources.tgz"
    for tar_name in [dataset_tar_name, resources_tar_name]:
        tar_path = target_dir / tar_name
        extracted_dir = corpus_dir / tar_name[:-4]
        completed_detector = extracted_dir / ".completed"
        if completed_detector.is_file():
            logging.info(f"Skip {tar_name} because {completed_detector} exists.")
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


def prepare_mobvoihotwords(
    corpus_dir: Pathlike, output_dir: Optional[Pathlike] = None
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions

    :param corpus_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'audio' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    dataset_parts = ["train", "dev", "test"]

    manifests = {}

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Maybe the manifests already exist: we can read them and save a bit of preparation time.
        manifests = read_manifests_if_cached(
            dataset_parts=dataset_parts, output_dir=output_dir
        )

    for part in dataset_parts:
        logging.info(f"Preparing MobvoiHotwords subset: {part}")
        if manifests_exist(part=part, output_dir=output_dir):
            logging.info(f"MobvoiHotwords subset: {part} already prepared - skipping.")
            continue
        # Generate a mapping: utt_id -> (audio_path, audio_info, speaker, text)
        recordings = []
        supervisions = []
        for prefix in ["p_", "n_"]:
            prefixed_part = prefix + part
            json_path = (
                corpus_dir
                / "mobvoi_hotword_dataset_resources"
                / f"{prefixed_part}.json"
            )
            with open(json_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
                for entry in json_data:
                    idx = entry["utt_id"]
                    speaker = (
                        idx if entry["speaker_id"] is None else entry["speaker_id"]
                    )
                    audio_path = corpus_dir / "mobvoi_hotword_dataset" / f"{idx}.wav"
                    text = "FREETEXT"
                    if entry["keyword_id"] == 0:
                        text = "HiXiaowen"
                    elif entry["keyword_id"] == 1:
                        text = "NihaoWenwen"
                    else:
                        assert entry["keyword_id"] == -1
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
