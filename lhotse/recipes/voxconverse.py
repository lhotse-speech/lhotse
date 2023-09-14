"""
VoxConverse is an audio-visual diarisation dataset consisting of multispeaker clips of human speech, extracted from YouTube videos.
Updates and additional information about the dataset can be found at our website (https://www.robots.ox.ac.uk/~vgg/data/voxconverse/index.html).
"""

import json
import logging
import re
import shutil
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, Optional, Union
from urllib.error import HTTPError

from tqdm.auto import tqdm

from lhotse import fix_manifests, validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import AlignmentItem, SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, is_module_available, resumable_download

DEV_AUDIO_ZIP = (
    "https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_dev_wav.zip"
)
TEST_AUDIO_ZIP = (
    "https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_test_wav.zip"
)
ANNOTATIONS_ZIP = "https://github.com/joonson/voxconverse/archive/master.zip"


def download_voxconverse(
    corpus_dir: Pathlike,
    force_download: bool = False,
):
    corpus_dir = Path(corpus_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    completed_detector = corpus_dir / ".completed"

    if not completed_detector.is_file() or force_download:
        print("Downloading VoxConverse dev set")
        resumable_download(DEV_AUDIO_ZIP, corpus_dir / "dev.zip")
        with zipfile.ZipFile(corpus_dir / "dev.zip") as zip_f:
            zip_f.extractall(corpus_dir / "dev")

        shutil.copytree(
            corpus_dir / "dev/audio", corpus_dir / "dev", dirs_exist_ok=True
        )
        shutil.rmtree(corpus_dir / "dev/audio")

        print("Downloading VoxConverse test set")
        resumable_download(TEST_AUDIO_ZIP, corpus_dir / "test.zip")
        with zipfile.ZipFile(corpus_dir / "test.zip") as zip_f:
            zip_f.extractall(corpus_dir / "test")

        shutil.copytree(
            corpus_dir / "test/voxconverse_test_wav",
            corpus_dir / "test",
            dirs_exist_ok=True,
        )
        shutil.rmtree(corpus_dir / "test/voxconverse_test_wav")

        print("Downloading VoxConverse annotations")
        resumable_download(ANNOTATIONS_ZIP, corpus_dir / "annotations.zip")
        with zipfile.ZipFile(corpus_dir / "annotations.zip") as zip_f:
            zip_f.extractall(corpus_dir)

        shutil.copytree(
            corpus_dir / "voxconverse-master", corpus_dir, dirs_exist_ok=True
        )
        shutil.rmtree(corpus_dir / "voxconverse-master")

        # cleanup
        (corpus_dir / "dev.zip").unlink()
        (corpus_dir / "test.zip").unlink()
        (corpus_dir / "annotations.zip").unlink()
        completed_detector.touch()

    print("Done")


def prepare_voxconverse(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    split_test: bool = False,  # test part is larger than dev part - split it into dev and test by default
):
    corpus_dir = Path(corpus_dir).absolute()

    splits = {}
    if split_test:
        splits["train"] = sorted((corpus_dir / "dev").glob("*.wav"))
        test_files = sorted((corpus_dir / "test").glob("*.wav"))
        splits["dev"] = test_files[: len(test_files) // 2]
        splits["test"] = test_files[len(test_files) // 2 :]
    else:
        splits["dev"] = sorted((corpus_dir / "dev").glob("*.wav"))
        splits["test"] = sorted((corpus_dir / "test").glob("*.wav"))

    manifests = {}
    for subset, wavs in splits.items():
        recordings = []
        supervisions = []
        for wav_file in wavs:
            recordings.append(Recording.from_file(wav_file))
            rttm_file = wav_file.with_suffix("").with_suffix(".rttm")
            for ix, (start, duration, speaker) in enumerate(_read_rttm(rttm_file)):
                supervisions.append(
                    SupervisionSegment(
                        id=f"{wav_file.stem}-{ix}",
                        recording_id=wav_file.stem,
                        start=start,
                        duration=duration,
                        channel=0,
                        language="en",
                        speaker=speaker,
                    )
                )

        recording_set = RecordingSet.from_recordings(recordings)
        supervision_set = SupervisionSet.from_segments(supervisions)
        recording_set, supervision_set = fix_manifests(recording_set, supervision_set)
        validate_recordings_and_supervisions(recording_set, supervision_set)

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            recording_set.to_file(
                output_dir / f"voxconverse_recordings_{subset}.jsonl.gz"
            )
            supervision_set.to_file(
                output_dir / f"voxconverse_supervisions_{subset}.jsonl.gz"
            )

        manifests[subset] = {
            "recordings": recording_set,
            "supervisions": supervision_set,
        }

    return manifests


def _read_rttm(filename):
    with open(filename, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if line.startswith("SPEAKER"):
                _, _, _, start, duration, _, _, speaker, _, _ = line.split()
                yield float(start), float(duration), speaker
