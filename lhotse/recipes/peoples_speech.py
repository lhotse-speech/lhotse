"""
The People’s Speech Dataset is among the world’s largest English speech recognition corpus today
that is licensed for academic and commercial usage under CC-BY-SA and CC-BY 4.0.
It includes 30,000+ hours of transcribed speech in English languages with a diverse set of speakers.
This open dataset is large enough to train speech-to-text systems and crucially is available with
a permissive license.
Just as ImageNet catalyzed machine learning for vision, the People’s Speech will unleash innovation
in speech research and products that are available to users across the globe.

Source: https://mlcommons.org/en/peoples-speech/
Full paper: https://openreview.net/pdf?id=R8CwidgJ0yT
"""

import logging
from collections import defaultdict
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from tqdm.auto import tqdm

from lhotse.audio import AudioSource, Recording, RecordingSet, info
from lhotse.qa import fix_manifests, validate_recordings_and_supervisions
from lhotse.recipes.utils import manifests_exist
from lhotse.serialization import load_jsonl
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, compute_num_samples

PEOPLES_SPEECH = (
    "train/dirty_sa",
    "train/dirty",
    "train/clean_sa",
    "train/clean",
    "validation/validation",
    "test/test",
)


def _parse_utterance(
    audio_dir: Pathlike,
    text: str,
    audio_path: str,
    identifier: str,
) -> Optional[Tuple[Recording, SupervisionSegment]]:
    full_path = audio_dir / audio_path

    recording = Recording.from_file(
        path=full_path,
        recording_id=full_path.stem,
    )
    segment = SupervisionSegment(
        id=recording.id,
        recording_id=recording.id,
        start=0.0,
        duration=recording.duration,
        channel=0,
        text=text,
        language="English",
        custom={"session_id": identifier},
    )

    return recording, segment


def _prepare_subset(
    subset: str,
    corpus_dir: Pathlike,
    num_jobs: int = 1,
) -> Tuple[RecordingSet, SupervisionSet]:
    """
    Returns the RecodingSet and SupervisionSet given a dataset part.
    :param subset: str, the name of the subset.
    :param corpus_dir: Pathlike, the path of the data dir.
    :return: the RecodingSet and SupervisionSet for train and valid.
    """
    corpus_dir = Path(corpus_dir)
    part_dir = corpus_dir / subset.split("/")[0]
    part_name = subset.split("/")[1]
    audio_dir = corpus_dir / subset

    with ThreadPoolExecutor(num_jobs) as ex:
        futures = []
        recordings = []
        supervisions = []
        for item in tqdm(
            # Note: People's Speech manifest.json is really a JSONL.
            load_jsonl(part_dir / f"{part_name}.json"),
            desc="Distributing tasks",
        ):
            for _, text, audio_path in zip(*item["training_data"].values()):
                futures.append(
                    ex.submit(
                        _parse_utterance,
                        audio_dir,
                        text,
                        audio_path,
                        item["identifier"],
                    )
                )

        for future in tqdm(futures, desc="Processing"):
            result = future.result()
            recording, segment = result
            recordings.append(recording)
            supervisions.append(segment)

        recording_set = RecordingSet.from_recordings(recordings)
        supervision_set = SupervisionSet.from_segments(supervisions)

        recording_set, supervision_set = fix_manifests(recording_set, supervision_set)
        validate_recordings_and_supervisions(
            recordings=recording_set, supervisions=supervision_set
        )

    return recording_set, supervision_set


def prepare_peoples_speech(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    num_jobs: int = 1,
) -> Dict[str, Union[RecordingSet, SupervisionSet]]:
    """
    Prepare :class:`~lhotse.RecordingSet` and :class:`~lhotse.SupervisionSet` manifests
    for The People's Speech.

    :param corpus_dir: Pathlike, the path of the main data dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a dict with keys "recordings" and "supervisions" with lazily opened manifests.
    """
    corpus_dir = Path(corpus_dir)

    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    logging.info("Preparing People's Speech...")

    subsets = PEOPLES_SPEECH

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    manifests = defaultdict(dict)

    for part in tqdm(subsets, desc="Dataset parts"):
        logging.info(f"Processing People's Speech subset: {part.split('/')[1]}")
        if manifests_exist(
            part=part.split("/")[1],
            output_dir=output_dir,
            prefix="peoples_speech",
            suffix="jsonl.gz",
        ):
            logging.info(
                f"People's Speech subset: {part.split('/')[1]} already prepared - skipping."
            )
            continue

        recording_set, supervision_set = _prepare_subset(part, corpus_dir, num_jobs)

        if output_dir is not None:
            supervision_set.to_file(
                output_dir
                / f"peoples_speech_supervisions_{part.split('/')[1]}.jsonl.gz"
            )
            recording_set.to_file(
                output_dir / f"peoples_speech_recordings_{part.split('/')[1]}.jsonl.gz"
            )

        manifests[part] = {"recordings": recording_set, "supervisions": supervision_set}

    return manifests
