"""
About the Bengali.AI Speech corpus

The competition dataset comprises about 1200 hours of recordings of Bengali speech.
Your goal is to transcribe recordings of speech that is out-of-distribution with respect to the training set.

Note that this is a Code Competition, in which the actual test set is hidden.
In this public version, we give some sample data in the correct format to help you author your solutions.
The full test set contains about 20 hours of speech in almost 8000 MP3 audio files.
All of the files in the test set are encoded at a sample rate of 32k, a bit rate of 48k, in one channel.

It is covered in more detail at https://arxiv.org/abs/2305.09688

Please download manually by
kaggle competitions download -c bengaliai-speech
"""

import logging
import os
from collections import defaultdict
from concurrent.futures.process import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from tqdm.auto import tqdm

from lhotse import (
    get_ffmpeg_torchaudio_info_enabled,
    set_ffmpeg_torchaudio_info_enabled,
)
from lhotse.audio import Recording, RecordingSet
from lhotse.qa import fix_manifests, validate_recordings_and_supervisions
from lhotse.recipes.utils import manifests_exist
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike

BENGALIAI_SPEECH = ("train", "valid", "test")


@contextmanager
def disable_ffmpeg_torchaudio_info() -> None:
    enabled = get_ffmpeg_torchaudio_info_enabled()
    set_ffmpeg_torchaudio_info_enabled(False)
    try:
        yield
    finally:
        set_ffmpeg_torchaudio_info_enabled(enabled)


def _parse_utterance(
    corpus_dir: Pathlike,
    audio_path: Pathlike,
    audio_id: str,
    text: Optional[str] = "",
) -> Optional[Tuple[Recording, SupervisionSegment]]:
    audio_path = audio_path.resolve()

    if not audio_path.is_file():
        logging.warning(f"No such file: {audio_path}")
        return None

    recording = Recording.from_file(
        path=audio_path,
        recording_id=audio_id,
    )
    segment = SupervisionSegment(
        id=audio_id,
        recording_id=audio_id,
        text=text,
        start=0.0,
        duration=recording.duration,
        channel=0,
        language="Bengali",
    )

    return recording, segment


def _prepare_subset(
    subset: str,
    corpus_dir: Pathlike,
    audio_info: Optional[dict] = None,
    num_jobs: int = 1,
) -> Tuple[RecordingSet, SupervisionSet]:
    """
    Returns the RecodingSet and SupervisionSet given a dataset part.
    :param subset: str, the name of the subset.
    :param corpus_dir: Pathlike, the path of the data dir.
    :return: the RecodingSet and SupervisionSet for train and valid.
    """
    corpus_dir = Path(corpus_dir)
    if subset == "test":
        part_path = corpus_dir / "test_mp3s"
    else:
        part_path = corpus_dir / "train_mp3s"

    audio_paths = list(part_path.rglob("*.mp3"))

    with disable_ffmpeg_torchaudio_info():
        with ProcessPoolExecutor(num_jobs) as ex:
            futures = []
            recordings = []
            supervisions = []
            for audio_path in tqdm(audio_paths, desc="Distributing tasks"):
                audio_id = os.path.split(str(audio_path))[1].replace(".mp3", "")
                if audio_info is not None:
                    if audio_id not in audio_info.keys():
                        continue
                    text = audio_info[audio_id]
                else:
                    text = None
                futures.append(
                    ex.submit(_parse_utterance, corpus_dir, audio_path, audio_id, text)
                )

            for future in tqdm(futures, desc="Processing"):
                result = future.result()
                if result is None:
                    continue
                recording, segment = result
                recordings.append(recording)
                supervisions.append(segment)

            recording_set = RecordingSet.from_recordings(recordings)
            supervision_set = SupervisionSet.from_segments(supervisions)

    return recording_set, supervision_set


def prepare_bengaliai_speech(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Path to the Bengali.AI Speech dataset.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'recordings' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)

    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    logging.info("Preparing Bengali.AI Speech...")

    subsets = BENGALIAI_SPEECH

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    manifests = defaultdict(dict)

    with open(corpus_dir / "train.csv") as f:
        audio_info = f.read().splitlines()

    train_info = {}
    valid_info = {}
    for line in audio_info[1:]:
        if ",train" in line:
            line = line.replace(",train", "").split(",", 1)
            train_info[line[0]] = line[1]
        elif ",valid" in line:
            line = line.replace(",valid", "").split(",", 1)
            valid_info[line[0]] = line[1]

    for part in tqdm(subsets, desc="Dataset parts"):
        logging.info(f"Processing Bengali.AI Speech subset: {part}")
        if manifests_exist(
            part=part,
            output_dir=output_dir,
            prefix="bengaliai_speech",
            suffix="jsonl.gz",
        ):
            logging.info(
                f"Bengali.AI Speech subset: {part} already prepared - skipping."
            )
            continue

        recording_set, supervision_set = _prepare_subset(
            subset=part,
            corpus_dir=corpus_dir,
            audio_info=train_info
            if part == "train"
            else valid_info
            if part == "valid"
            else None,
            num_jobs=num_jobs,
        )

        # Fix manifests
        recording_set, supervision_set = fix_manifests(recording_set, supervision_set)
        validate_recordings_and_supervisions(recording_set, supervision_set)

        if output_dir is not None:
            supervision_set.to_file(
                output_dir / f"bengaliai_speech_supervisions_{part}.jsonl.gz"
            )
            recording_set.to_file(
                output_dir / f"bengaliai_speech_recordings_{part}.jsonl.gz"
            )

        manifests[part] = {"recordings": recording_set, "supervisions": supervision_set}

    return manifests
