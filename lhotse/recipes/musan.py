"""
This script creates the MUSAN data directory.
Consists of babble, music and noise files.
Used to create augmented data
The required dataset is freely available at http://www.openslr.org/17/

The corpus can be cited as follows:
@misc{musan2015,
 author = {David Snyder and Guoguo Chen and Daniel Povey},
 title = {{MUSAN}: {A} {M}usic, {S}peech, and {N}oise {C}orpus},
 year = {2015},
 eprint = {1510.08484},
 note = {arXiv:1510.08484v1}
}
"""
import logging
import tarfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

from lhotse import (
    Recording,
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
    validate,
    validate_recordings_and_supervisions,
)
from lhotse.utils import Pathlike, urlretrieve_progress

MUSAN_URL = "https://www.openslr.org/resources/17/musan.tar.gz"


def download_musan(
    target_dir: Pathlike = ".",
    url: Optional[str] = MUSAN_URL,
    force_download: Optional[bool] = False,
) -> Path:
    """
    Download and untar the MUSAN corpus.

    :param target_dir: Pathlike, the path of the dir to store the dataset.
    :param url: str, the url that downloads file called "musan.tar.gz".
    :param force_download: bool, if True, download the archive even if it already exists.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    tar_name = "musan.tar.gz"
    tar_path = target_dir / tar_name
    corpus_dir = target_dir / "musan"
    completed_detector = target_dir / ".musan_completed"
    if completed_detector.is_file():
        logging.info(f"Skipping {tar_name} because {completed_detector} exists.")
        return corpus_dir
    if force_download or not tar_path.is_file():
        urlretrieve_progress(url, filename=tar_path, desc=f"Downloading {tar_name}")
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=target_dir)
        completed_detector.touch()
    return corpus_dir


def prepare_musan(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    parts: Sequence[str] = ("music", "speech", "noise"),
    use_vocals: bool = True,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    if not parts:
        raise ValueError("No MUSAN parts specified for manifest preparation.")
    if isinstance(parts, str):
        parts = [parts]

    manifests = {}
    if "music" in parts:
        manifests["music"] = prepare_music(corpus_dir, use_vocals=use_vocals)
        validate_recordings_and_supervisions(**manifests["music"])
    if "speech" in parts:
        manifests["speech"] = {"recordings": scan_recordings(corpus_dir / "speech")}
        validate(manifests["speech"]["recordings"])
    if "noise" in parts:
        manifests["noise"] = {"recordings": scan_recordings(corpus_dir / "noise")}
        validate(manifests["noise"]["recordings"])

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for part in manifests:
            for key, manifest in manifests[part].items():
                manifest.to_json(output_dir / f"{key}_{part}.json")

    return manifests


def prepare_music(
    corpus_dir: Path, use_vocals: bool = True
) -> Dict[str, Union[RecordingSet, SupervisionSet]]:
    music_dir = corpus_dir / "music"
    recordings = scan_recordings(music_dir)
    supervisions = SupervisionSet.from_segments(
        SupervisionSegment(
            id=utt,
            recording_id=utt,
            start=0,
            duration=recordings.duration(utt),
            speaker=musician,
            custom={"genres": genres.split(","), "vocals": vocals == "Y"},
        )
        for file in music_dir.rglob("ANNOTATIONS")
        for utt, genres, vocals, musician in read_annotations(file, max_fields=4)
    )
    if not use_vocals:
        supervisions = supervisions.filter(lambda s: s.custom["vocals"] is False)
    return {"recordings": recordings, "supervisions": supervisions}


def scan_recordings(corpus_dir: Path) -> RecordingSet:
    return RecordingSet.from_recordings(
        Recording.from_file(file) for file in corpus_dir.rglob("*.wav")
    )


def read_annotations(
    path: Path, max_fields: Optional[int] = None
) -> Iterable[List[str]]:
    with open(path) as f:
        for line in f:
            line = line.strip().split()
            if line:
                yield line if max_fields is None else line[:max_fields]
