"""
WHAM noise recordings preparation for Lhotse.

This recipe prepares the noise component of the WHAM dataset, which consists
of real-world ambient noise recordings collected in various environments
(cafes, restaurants, bars, etc.). These noise recordings are commonly used
in combination with clean speech datasets to create noisy mixtures for
speech separation and enhancement tasks.

For more details about WHAM, see:
- Paper: "WHAM!: Extending Speech Separation to Noisy Environments"
  https://arxiv.org/abs/1907.01160
- Original dataset: https://wham.csail.mit.edu/
"""
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union
from zipfile import ZipFile

from lhotse import Recording, RecordingSet, SupervisionSet, validate
from lhotse.utils import Pathlike, resumable_download, safe_extract

WHAM_URL = (
    "https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip"
)


def download_wham(
    target_dir: Pathlike = ".",
    url: Optional[str] = WHAM_URL,
    force_download: Optional[bool] = False,
) -> Path:
    """
    Download and untar the WHAM corpus.

    :param target_dir: Pathlike, the path of the dir to store the dataset.
    :param url: str, the url that downloads file called "wham_noise.zip".
    :param force_download: bool, if True, download the archive even if it already exists.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    zip_name = "wham_noise.zip"
    zip_path = target_dir / zip_name
    corpus_dir = target_dir / "wham_noise"
    completed_detector = target_dir / ".wham_noise_completed"
    if completed_detector.is_file():
        logging.info(f"Skipping {zip_name} because {completed_detector} exists.")
        return corpus_dir
    resumable_download(url, filename=zip_path, force_download=force_download)
    logging.info("Extracting files...")
    with ZipFile(zip_path) as zf:
        zf.extractall(path=target_dir)
        completed_detector.touch()
    return corpus_dir


def prepare_wham(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    manifests = {}
    splits = ["tr", "cv", "tt"]

    for split in splits:
        logging.info(f"Scanning {split} split...")
        manifests[split] = {"recordings": scan_recordings(corpus_dir / split)}
        logging.info(f"Validating {split} split...")
        validate(manifests[split]["recordings"])

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for split in manifests:
            for key, manifest in manifests[split].items():
                manifest.to_file(output_dir / f"wham_{key}_{split}.jsonl.gz")

    return manifests


def scan_recordings(corpus_dir: Path) -> RecordingSet:
    return RecordingSet.from_recordings(
        Recording.from_file(file) for file in corpus_dir.rglob("*.wav")
    )
