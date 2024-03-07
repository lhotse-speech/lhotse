"""
The following are the original TED-LIUM 2 README contents.

This is theTED-LIUM corpus release 2, English speech recognition training corpus from TED talks, created by Laboratoire d’Informatique de l’Université du Maine (LIUM) (mirrored here)
licensed under Creative Commons BY-NC-ND 3.0 (http://creativecommons.org/licenses/by-nc-nd/3.0/deed.en).

All talks and text are property of TED Conferences LLC.

--- 

The TED-LIUM corpus was made from audio talks and their transcriptions available on the TED website. We have prepared and filtered these data in order to train acoustic models to participate to the International Workshop on Spoken Language Translation 2011 (the LIUM English/French SLT system reached the first rank in the SLT task). 

More details are given in this paper: 

A. Rousseau, P. Deléglise, and Y. Estève, "Enhancing the TED-LIUM Corpus with Selected Data for Language Modeling and More TED Talks",
in Proceedings of the Ninth International Conference on Language Resources and Evaluation (LREC’14), May 2014.


Please cite this reference if you use these data in your research work. 

--- 

Contents: 

- 1495 audio talks in NIST sphere format (SPH) 
- 1495 transcripts in STM format 
- Dictionary with pronunciation (159848 entries) 
- Selected monolingual data for language modeling from WMT12 publicly available corpora


SPH format info: 

Channels			: 1
Sample Rate		: 16000
Precision			: 16-bit
Bit Rate			: 256k
Sample Encoding	: 16-bit Signed Integer PCM

"""

import logging
import shutil
import tarfile
from concurrent.futures.thread import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

from lhotse import (
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
    validate_recordings_and_supervisions,
)
from lhotse.qa import fix_manifests
from lhotse.recipes.utils import normalize_text_tedlium
from lhotse.utils import Pathlike, resumable_download, safe_extract

TEDLIUM_PARTS = ("train", "dev", "test")


def download_tedlium2(
    target_dir: Pathlike = ".", force_download: Optional[bool] = False
) -> Path:
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    tar_path = target_dir / "TEDLIUM_release2.tar.gz"
    corpus_dir = target_dir / "TEDLIUM_release2"
    completed_detector = corpus_dir / ".completed"
    if completed_detector.is_file():
        logging.info(f"Skipping {tar_path.name} because {completed_detector} exists.")
        return corpus_dir
    resumable_download(
        "http://www.openslr.org/resources/19/TEDLIUM_release2.tar.gz",
        filename=tar_path,
        force_download=force_download,
    )
    shutil.rmtree(corpus_dir, ignore_errors=True)
    with tarfile.open(tar_path) as tar:
        safe_extract(tar, path=target_dir)
    completed_detector.touch()
    return corpus_dir


def prepare_tedlium2(
    tedlium_root: Pathlike,
    output_dir: Optional[Pathlike] = None,
    dataset_parts: Union[str, Sequence[str]] = TEDLIUM_PARTS,
    num_jobs: int = 1,
    normalize_text: str = "none",
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Prepare manifests for the TED-LIUM v2 corpus.

    The manifests are created in a dict with three splits: train, dev and test.
    Each split contains a RecordingSet and SupervisionSet in a dict under keys 'recordings' and 'supervisions'.

    :param tedlium_root: Path to the unpacked TED-LIUM data.
    :param output_dir: Path where the manifests should be written.
    :param dataset_parts: Which parts of the dataset to prepare.
        By default, all parts are prepared.
    :param num_jobs: Number of parallel jobs to use.
    :return: A dict with standard corpus splits containing the manifests.
    """
    tedlium_root = Path(tedlium_root)
    output_dir = Path(output_dir) if output_dir is not None else None
    corpus = {}

    dataset_parts = [dataset_parts] if isinstance(dataset_parts, str) else dataset_parts

    with ThreadPoolExecutor(num_jobs) as ex:
        for split in dataset_parts:
            logging.info(f"Processing {split} split...")
            root = tedlium_root / split
            recordings = RecordingSet.from_dir(
                root / "sph", pattern="*.sph", num_jobs=num_jobs
            )
            stms = list((root / "stm").glob("*.stm"))
            assert len(stms) == len(recordings), (
                f"Mismatch: found {len(recordings)} "
                f"sphere files and {len(stms)} STM files. "
                f"You might be missing some parts of TEDLIUM..."
            )
            futures = []
            _parse_stm_worker = partial(_parse_stm_file, normalize_text=normalize_text)
            for stm in stms:
                futures.append(ex.submit(_parse_stm_worker, stm))

            segments = []
            for future in futures:
                segments.extend(future.result())

            supervisions = SupervisionSet.from_segments(segments)
            recordings, supervisions = fix_manifests(recordings, supervisions)

            corpus[split] = {"recordings": recordings, "supervisions": supervisions}
            validate_recordings_and_supervisions(**corpus[split])

            if output_dir is not None:
                recordings.to_file(output_dir / f"tedlium2_recordings_{split}.jsonl.gz")
                supervisions.to_file(
                    output_dir / f"tedlium2_supervisions_{split}.jsonl.gz"
                )

    return corpus


def _parse_stm_file(stm: str, normalize_text: str = "none") -> SupervisionSegment:
    """Helper function to parse a single STM file."""
    segments = []
    with stm.open() as f:
        for idx, l in enumerate(f):
            rec_id, _, _, start, end, _, *words = l.split()
            start, end = float(start), float(end)
            text = " ".join(words).replace("{NOISE}", "[NOISE]")
            if text == "ignore_time_segment_in_scoring":
                continue
            segments.append(
                SupervisionSegment(
                    id=f"{rec_id}-{idx}",
                    recording_id=rec_id,
                    start=start,
                    duration=round(end - start, ndigits=8),
                    channel=0,
                    text=normalize_text_tedlium(text, normalize_text),
                    language="English",
                    speaker=rec_id,
                )
            )
    return segments
