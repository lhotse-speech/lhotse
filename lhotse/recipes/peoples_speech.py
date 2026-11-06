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
from concurrent.futures.thread import ThreadPoolExecutor
from itertools import islice
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple, Union

from tqdm.auto import tqdm

from lhotse.audio import AudioSource, Recording, RecordingSet, info
from lhotse.cut import CutSet
from lhotse.qa import fix_manifests, validate_recordings_and_supervisions
from lhotse.recipes.utils import (
    create_cuts_per_recording,
    manifests_exist,
    read_manifests_if_cached,
)
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


def _iter_subset(
    subset: str,
    corpus_dir: Pathlike,
    num_jobs: int = 1,
    chunk_size: int = 1000,
) -> Iterator[Tuple[Recording, SupervisionSegment]]:
    """
    Yields the (Recording, SupervisionSegment) pairs of a dataset part,
    reading the audio headers in chunks to keep the memory usage bounded.
    :param subset: str, the name of the subset, e.g. ``"train/clean"``.
    :param corpus_dir: Pathlike, the path of the data dir.
    """
    corpus_dir = Path(corpus_dir)
    part_dir = corpus_dir / subset.split("/")[0]
    part_name = subset.split("/")[1]
    audio_dir = corpus_dir / subset

    def utterances():
        # Note: People's Speech manifest.json is really a JSONL.
        for item in load_jsonl(part_dir / f"{part_name}.json"):
            for _, text, audio_path in zip(*item["training_data"].values()):
                yield audio_dir, text, audio_path, item["identifier"]

    utts = utterances()
    with ThreadPoolExecutor(num_jobs) as ex:
        while True:
            chunk = list(islice(utts, chunk_size))
            if not chunk:
                break
            futures = [ex.submit(_parse_utterance, *args) for args in chunk]
            for future in futures:
                yield future.result()


def prepare_peoples_speech(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet, CutSet]]]:
    """
    Prepare :class:`~lhotse.RecordingSet`, :class:`~lhotse.SupervisionSet`,
    and :class:`~lhotse.CutSet` manifests for The People's Speech.

    When ``output_dir`` is given, the manifests are streamed to disk in a single pass
    (which skips :func:`~lhotse.qa.fix_manifests`) and returned lazily.
    Parts that are already present in ``output_dir`` are read instead of being prepared again.

    :param corpus_dir: Pathlike, the path of the main data dir.
    :param output_dir: Pathlike, the path where to write the manifests. Can be omitted to avoid writing.
    :return: a Dict whose key is the dataset part (``clean``, ``clean_sa``, ``dirty``, ``dirty_sa``,
        ``validation``, ``test``), and the value is a Dict with the keys "recordings", "supervisions", and "cuts".
    """
    corpus_dir = Path(corpus_dir)

    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    logging.info("Preparing People's Speech...")

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    manifests = {}

    for subset in tqdm(PEOPLES_SPEECH, desc="Dataset parts"):
        part = subset.split("/")[1]
        logging.info(f"Processing People's Speech subset: {part}")
        if manifests_exist(part=part, output_dir=output_dir, prefix="peoples-speech"):
            logging.info(f"People's Speech subset: {part} already prepared - skipping.")
            manifests.update(
                read_manifests_if_cached(
                    dataset_parts=[part], output_dir=output_dir, prefix="peoples-speech"
                )
            )
            continue

        if output_dir is None:
            recordings, supervisions = zip(
                *_iter_subset(subset, corpus_dir, num_jobs=num_jobs)
            )
            recordings, supervisions = fix_manifests(
                RecordingSet.from_recordings(recordings),
                SupervisionSet.from_segments(supervisions),
            )
            validate_recordings_and_supervisions(recordings, supervisions)
            manifests[part] = {
                "recordings": recordings,
                "supervisions": supervisions,
                "cuts": create_cuts_per_recording(recordings, supervisions),
            }
            continue

        paths = {
            name: output_dir / f"peoples-speech_{name}_{part}.jsonl.gz"
            for name in ("recordings", "supervisions", "cuts")
        }
        with (
            RecordingSet.open_writer(
                paths["recordings"], overwrite=True
            ) as recs_writer,
            SupervisionSet.open_writer(
                paths["supervisions"], overwrite=True
            ) as sups_writer,
            CutSet.open_writer(paths["cuts"], overwrite=True) as cuts_writer,
        ):
            for recording, segment in _iter_subset(
                subset, corpus_dir, num_jobs=num_jobs
            ):
                recs_writer.write(recording)
                sups_writer.write(segment)
                cut = recording.to_cut()
                cut.supervisions = [segment]
                cuts_writer.write(cut)
        manifests[part] = {
            "recordings": RecordingSet.from_jsonl_lazy(paths["recordings"]),
            "supervisions": SupervisionSet.from_jsonl_lazy(paths["supervisions"]),
            "cuts": CutSet.from_jsonl_lazy(paths["cuts"]),
        }

    return manifests
