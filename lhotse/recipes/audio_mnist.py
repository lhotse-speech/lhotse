"""
About the AudioMNIST corpus

The AudioMNIST dataset consists of 30000 audio recordings
(ca. 9.5 hours) of spoken digits (0-9) in English with 50 repetitions
per digit for each of the 60 different speakers. Recordings were
collected in quiet offices with a RØDE NT-USB microphone as
mono channel signal at a sampling frequency of 48kHz and were
saved in 16 bit integer format. In addition to audio recordings, meta
information including age (range: 22 - 61 years), gender (12 female
/ 48 male), origin and accent of all speakers were collected as well.
All speakers were informed about the intent of the data collection
and have given written declarations of consent for their participation prior
to their recording session.


The data is publicly available at the following github repo:

    https://github.com/soerenab/AudioMNIST
"""

import logging
import os
import tarfile
from pathlib import Path
from typing import Dict, Union

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.qa import fix_manifests
from lhotse.serialization import load_json
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, resumable_download

# The last update is 5 years ago so it is safe to use the master branch.
AUDIO_MNIST_URL = "https://github.com/soerenab/AudioMNIST/archive/master.tar.gz"  # noqa


def download_audio_mnist(
    target_dir: Pathlike,
    force_download: bool = False,
) -> Path:
    """
    Download and untar the AudioMNIST dataset.

    :param target_dir: Pathlike, the path of the dir to store the dataset.
    :param force_download: bool, if True, download the archive even if it
      already exists.
    :return: Return the directory containing the extracted dataset.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    tgz_name = "master.tar.gz"
    tgz_path = target_dir / tgz_name
    if tgz_path.exists() and not force_download:
        logging.info(f"Skipping {tgz_name} because file exists.")

    resumable_download(
        AUDIO_MNIST_URL,
        tgz_path,
        force_download=force_download,
    )
    tgz_dir = target_dir / "AudioMNIST"
    if not tgz_dir.exists():
        logging.info(f"Untarring {tgz_name}.")
        with tarfile.open(tgz_path) as tar:
            tar.extractall(path=target_dir)
        os.rename(str(target_dir / "AudioMNIST-master"), str(tgz_dir))
    return tgz_dir


def prepare_audio_mnist(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """Prepare manifests for the AudioMNIST corpus.

    :param: corpus_dir: We assume it is the github repo directory and it
      contains the following directories: data/{01,02,03,...,60}
    :param: output_dir: Directory where the manifests should be written.
    """
    in_data_dir = Path(corpus_dir) / "data"
    assert (Path(in_data_dir) / "audioMNIST_meta.txt").is_file()

    metadata = load_json(in_data_dir / "audioMNIST_meta.txt")
    assert len(metadata) == 60, len(metadata)
    for i in range(1, 61):
        assert f"{i:02}" in metadata, i

    recordings = RecordingSet.from_recordings(
        Recording.from_file(
            path=path,
            # converts:
            #   path/to/AudioMNIST/data/01/3_01_24.wav
            # to:
            #   3_01_24
            recording_id=path.stem,
        )
        for i in range(1, 61)
        for path in (in_data_dir / f"{i:02}").rglob("*.wav")
    )

    id2text = {
        "0": "zero",
        "1": "one",
        "2": "two",
        "3": "three",
        "4": "four",
        "5": "five",
        "6": "six",
        "7": "seven",
        "8": "eight",
        "9": "nine",
    }

    supervisions = []
    for r in recordings:
        digit, speaker_id, _ = r.id.split("_")

        supervisions.append(
            SupervisionSegment(
                id=r.id,
                recording_id=r.id,
                start=0,
                duration=r.duration,
                channel=0,
                text=id2text[digit],
                language="English",
                speaker=speaker_id,
                custom=metadata[speaker_id],
            )
        )

    supervisions = SupervisionSet.from_segments(supervisions)
    recordings, supervisions = fix_manifests(recordings, supervisions)
    validate_recordings_and_supervisions(recordings, supervisions)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        supervisions.to_file(output_dir / "audio_mnist_supervisions.jsonl.gz")
        recordings.to_file(output_dir / "audio_mnist_recordings.jsonl.gz")

    return {"recordings": recordings, "supervisions": supervisions}
