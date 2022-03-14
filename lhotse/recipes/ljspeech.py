"""
The LJ Speech Dataset is a public domain speech dataset consisting of 13,100 short audio clips of a single speaker
reading passages from 7 non-fiction books. A transcription is provided for each clip. Clips vary in length from 1 to
10 seconds and have a total length of approximately 24 hours.

The texts were published between 1884 and 1964, and are in the public domain. The audio was recorded in 2016-17 by
the LibriVox project and is also in the public domain.

See https://keithito.com/LJ-Speech-Dataset for more details.
"""

import logging
import re
import shutil
import tarfile
from pathlib import Path
from typing import Dict, Optional, Union

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.features import Fbank
from lhotse.features.base import TorchaudioFeatureExtractor
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, fastcopy, urlretrieve_progress


def download_ljspeech(
    target_dir: Pathlike = ".", force_download: Optional[bool] = False
) -> Path:
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = "LJSpeech-1.1"
    tar_path = target_dir / f"{dataset_name}.tar.bz2"
    corpus_dir = target_dir / dataset_name
    completed_detector = corpus_dir / ".completed"
    if completed_detector.is_file():
        logging.info(f"Skipping {dataset_name} because {completed_detector} exists.")
        return corpus_dir
    if force_download or not tar_path.is_file():
        urlretrieve_progress(
            f"http://data.keithito.com/data/speech/{dataset_name}.tar.bz2",
            filename=tar_path,
            desc="Downloading LJSpeech",
        )
    shutil.rmtree(corpus_dir, ignore_errors=True)
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=target_dir)
    completed_detector.touch()

    return corpus_dir


def prepare_ljspeech(
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

    # Generate a mapping: utt_id -> (audio_path, audio_info, text)
    metadata_csv_path = corpus_dir / "metadata.csv"
    assert metadata_csv_path.is_file(), f"No such file: {metadata_csv_path}"
    recordings = []
    supervisions = []
    with open(metadata_csv_path) as f:
        for line in f:
            recording_id, text, _ = line.split("|")
            audio_path = corpus_dir / "wavs" / f"{recording_id}.wav"
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
                language="English",
                gender="female",
                text=text,
            )
            recordings.append(recording)
            supervisions.append(segment)

    recording_set = RecordingSet.from_recordings(recordings)
    supervision_set = SupervisionSet.from_segments(supervisions)

    validate_recordings_and_supervisions(recording_set, supervision_set)

    if output_dir is not None:
        supervision_set.to_json(output_dir / "supervisions.json")
        recording_set.to_json(output_dir / "recordings.json")

    return {"recordings": recording_set, "supervisions": supervision_set}


def feature_extractor() -> TorchaudioFeatureExtractor:
    """
    Set up the feature extractor for TTS task.
    :return: A feature extractor with custom parameters.
    """
    feature_extractor = Fbank()
    feature_extractor.config.num_mel_bins = 80

    return feature_extractor


def text_normalizer(segment: SupervisionSegment) -> SupervisionSegment:
    text = segment.text.upper()
    text = re.sub(r"[^\w !?]", "", text)
    text = re.sub(r"^\s+", "", text)
    text = re.sub(r"\s+$", "", text)
    text = re.sub(r"\s+", " ", text)
    return fastcopy(segment, text=text)
