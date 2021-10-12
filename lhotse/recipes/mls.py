"""
Multilingual LibriSpeech (MLS) dataset is a large multilingual corpus suitable for speech research.
The dataset is derived from read audiobooks from LibriVox and consists of 8 languages -
English, German, Dutch, Spanish, French, Italian, Portuguese, Polish.
It is available at OpenSLR: http://openslr.org/94
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict

from tqdm.auto import tqdm

from lhotse import *
from lhotse.utils import Pathlike


def prepare_mls(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    opus: bool = True,
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]]:
    """
    Prepare Multilingual LibriSpeech corpus.

    Returns a dict structured like the following:

    .. code-block:: python

        {
            'english': {
                'train': {'recordings': RecordingSet(...), 'supervisions': SupervisionSet(...)},
                'dev': ...,
                'test': ...
            },
            'polish': { ... },
            ...
        }

    :param corpus_dir: Path to the corpus root (directories with specific languages should be inside).
    :param output_dir: Optional path where the manifests should be stored.
    :param opus: Should we scan for OPUS files (otherwise we'll look for FLAC files).
    :param num_jobs: How many jobs should be used for creating recording manifests.
    :return: A dict with structure: ``d[language][split] = {recordings, supervisions}``.
    """
    corpus_dir = Path(corpus_dir)
    output_dir = Path(output_dir) if output_dir is not None else None
    assert corpus_dir.is_dir()

    languages = {
        d.name.split("_")[1]: d
        for d in corpus_dir.glob("mls_*")
        if d.is_dir() and "_lm_" not in d.name and (opus or not d.name.endswith("opus"))
    }
    logging.info(f"Found MLS languages: {list(languages)}")

    manifests = defaultdict(dict)
    for lang, lang_dir in tqdm(
        languages.items(), desc="Langauges", total=len(languages)
    ):
        logging.info(f"Processing language: {lang}")

        # Read the speaker to gender mapping.
        spk2gender = {}
        for line in (lang_dir / "metainfo.txt").read_text().splitlines():
            spk, gender, *_ = line.split("|")
            spk2gender[spk.strip()] = gender.strip()

        for split in tqdm(["test", "dev", "train"], desc="Splits"):

            # If everything is ready, read it and skip it.
            recordings_path = (
                None
                if output_dir is None
                else output_dir / f"recordings_{lang}_{split}.jsonl.gz"
            )
            supervisions_path = (
                None
                if output_dir is None
                else output_dir / f"supervisions_{lang}_{split}.jsonl.gz"
            )
            if (
                recordings_path is not None
                and recordings_path.is_file()
                and supervisions_path is not None
                and supervisions_path.is_file()
            ):
                logging.info(f"Skipping - {lang}/{split} - already exists!")
                recordings = RecordingSet.from_file(recordings_path)
                supervisions = SupervisionSet.from_file(supervisions_path)
                manifests[lang][split] = {
                    "recordings": recordings,
                    "supervisions": supervisions,
                }
                continue

            # Create recordings manifest.
            split_dir = lang_dir / split
            recordings = RecordingSet.from_dir(
                path=split_dir,
                pattern="*.opus" if opus else "*.flac",
                num_jobs=num_jobs,
                force_opus_sampling_rate=16000,
            )

            # Create supervisions manifest.
            supervisions = []
            for line in (split_dir / "transcripts.txt").read_text().splitlines():
                recording_id, text = line.split("\t")
                speaker = recording_id.split("_")[0]
                supervisions.append(
                    SupervisionSegment(
                        id=recording_id,
                        recording_id=recording_id,
                        text=text,
                        speaker=speaker,
                        gender=spk2gender[speaker],
                        start=0.0,
                        duration=recordings.duration(recording_id),
                        language=lang,
                    )
                )
            supervisions = SupervisionSet.from_segments(supervisions)

            # Fix any missing recordings/supervisions.
            recordings, supervisions = fix_manifests(recordings, supervisions)
            validate_recordings_and_supervisions(recordings, supervisions)

            # Save for return.
            manifests[lang][split] = {
                "recordings": recordings,
                "supervisions": supervisions,
            }

            # Optional storage on disk.
            if output_dir is not None:
                output_dir.mkdir(exist_ok=True, parents=True)
                recordings.to_jsonl(recordings_path)
                supervisions.to_jsonl(supervisions_path)

    return dict(manifests)
