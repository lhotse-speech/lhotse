"""
Hi-Fi Multi-Speaker English TTS Dataset (Hi-Fi TTS) is a multi-speaker English dataset
for training text-to-speech models.
The dataset is based on public audiobooks from LibriVox and texts from Project Gutenberg.
The Hi-Fi TTS dataset contains about 291.6 hours of speech from 10 speakers
with at least 17 hours per speaker sampled at 44.1 kHz.

For more information and the latest dataset statistics, please refer to the paper:
"Hi-Fi Multi-Speaker English TTS Dataset" Bakhturina, E., Lavrukhin, V., Ginsburg, B.
and Zhang, Y., 2021: arxiv.org/abs/2104.01497.

BibTeX entry for citations:

@article{bakhturina2021hi,
  title={{Hi-Fi Multi-Speaker English TTS Dataset}},
  author={Bakhturina, Evelina and Lavrukhin, Vitaly and Ginsburg, Boris and Zhang, Yang},
  journal={arXiv preprint arXiv:2104.01497},
  year={2021}
}
"""
import logging
import shutil
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional, Union

from tqdm.auto import tqdm

from lhotse import (
    Recording,
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
    validate_recordings_and_supervisions,
)
from lhotse.recipes.utils import manifests_exist, read_manifests_if_cached
from lhotse.serialization import load_jsonl
from lhotse.utils import Pathlike, urlretrieve_progress

ID2SPEAKER = {
    "92": "Cori Samuel",
    "6097": "Phil Benson",
    "9017": "John Van Stan",
    "6670": "Mike Pelton",
    "6671": "Tony Oliva",
    "8051": "Maria Kasper",
    "9136": "Helen Taylor",
    "11614": "Sylviamb",
    "11697": "Celine Major",
    "12787": "LikeManyWaters",
}

ID2GENDER = {
    "92": "F",
    "6097": "M",
    "9017": "M",
    "6670": "M",
    "6671": "M",
    "8051": "F",
    "9136": "F",
    "11614": "F",
    "11697": "F",
    "12787": "F",
}


def download_hifitts(
    target_dir: Pathlike = ".",
    force_download: Optional[bool] = False,
    base_url: Optional[str] = "http://www.openslr.org/resources",
) -> Path:
    """
    Download and untar the HiFi TTS dataset.

    :param target_dir: Pathlike, the path of the dir to store the dataset.
    :param force_download: Bool, if True, download the tars no matter if the tars exist.
    :param base_url: str, the url of the OpenSLR resources.
    :return: the path to downloaded and extracted directory with data.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    url = f"{base_url}/109"
    tar_name = "hi_fi_tts_v0.tar.gz"
    tar_path = target_dir / tar_name
    part_dir = target_dir / f"hi_fi_tts_v0"
    completed_detector = part_dir / ".completed"
    if completed_detector.is_file():
        logging.info(
            f"Skipping HiFiTTS preparation because {completed_detector} exists."
        )
        return part_dir
    if force_download or not tar_path.is_file():
        urlretrieve_progress(
            f"{url}/{tar_name}", filename=tar_path, desc=f"Downloading {tar_name}"
        )
    shutil.rmtree(part_dir, ignore_errors=True)
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=target_dir)
    completed_detector.touch()

    return part_dir


def prepare_hifitts(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Prepare manifests for the HiFiTTS dataset.

    :param corpus_dir: Path or str, the path to the downloaded corpus main directory.
    :param output_dir: Path or str, the path where to write the manifests.
    :param num_jobs: How many concurrent workers to use for preparing each dataset partition.
    :return: a dict with manifests for all the partitions
        (example query: ``manifests['92_clean_train']['recordings']``).
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    manifests = {}

    json_manifests = list(corpus_dir.glob("*.json"))
    dataset_partitions = [to_partition_id(p) for p in json_manifests]

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Maybe the manifests already exist: we can read them and save a bit of preparation time.
        manifests = read_manifests_if_cached(
            dataset_parts=dataset_partitions, output_dir=output_dir, prefix="hifitts"
        )

    with ProcessPoolExecutor(num_jobs) as ex:
        futures = []
        partition_ids = []
        for raw_manifest_path in json_manifests:
            speaker_id, _, clean_or_other, part = raw_manifest_path.stem.split("_")
            partition_id = to_partition_id(raw_manifest_path)
            if manifests_exist(
                part=partition_id, output_dir=output_dir, prefix="hifitts"
            ):
                logging.info(f"HiFiTTS subset: {part} already prepared - skipping.")
                continue
            futures.append(
                ex.submit(
                    prepare_single_partition,
                    raw_manifest_path,
                    corpus_dir,
                    speaker_id,
                    clean_or_other,
                )
            )
            partition_ids.append(partition_id)

        for future, partition_id in tqdm(
            zip(as_completed(futures), partition_ids),
            desc="Preparing HiFiTTS parts",
            total=len(futures),
        ):
            recordings, supervisions = future.result()

            if output_dir is not None:
                supervisions.to_json(
                    output_dir / f"hifitts_supervisions_{partition_id}.json"
                )
                recordings.to_json(
                    output_dir / f"hifitts_recordings_{partition_id}.json"
                )

            manifests[partition_id] = {
                "recordings": recordings,
                "supervisions": supervisions,
            }

    return manifests


def prepare_single_partition(
    raw_manifest_path: Path,
    corpus_dir: Path,
    speaker_id: str,
    clean_or_other: str,
):
    recordings = []
    supervisions = []
    for meta in load_jsonl(raw_manifest_path):
        recording = Recording.from_file(corpus_dir / meta["audio_filepath"])
        recordings.append(recording)
        supervisions.append(
            SupervisionSegment(
                id=recording.id,
                recording_id=recording.id,
                start=0,
                duration=recording.duration,
                channel=0,
                text=meta["text"],
                speaker=ID2SPEAKER[speaker_id],
                gender=ID2GENDER[speaker_id],
                custom={"text_punct": meta["text_normalized"], "split": clean_or_other},
            )
        )
    recordings = RecordingSet.from_recordings(recordings)
    supervisions = SupervisionSet.from_segments(supervisions)
    validate_recordings_and_supervisions(recordings, supervisions)
    return recordings, supervisions


def to_partition_id(path: Path) -> str:
    speaker_id, _, clean_or_other, part = path.stem.split("_")
    return f"{speaker_id}_{clean_or_other}_{part}"
