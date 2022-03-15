"""
The following are the original TED-LIUM 3 README contents.

This is the TED-LIUM corpus release 3,
licensed under Creative Commons BY-NC-ND 3.0 (http://creativecommons.org/licenses/by-nc-nd/3.0/deed.en).

All talks and text are property of TED Conferences LLC.

This new TED-LIUM release was made through a collaboration between the Ubiqus company and the LIUM (University of Le Mans, France)

---

Contents:

- 2351 audio talks in NIST sphere format (SPH), including talks from TED-LIUM 2: be careful, same talks but not same audio files (only these audio file must be used with the TED-LIUM 3 STM files)
--> 452 hours of audio
- 2351 aligned automatic transcripts in STM format

- TEDLIUM 2 dev and test data: 19 TED talks in SPH format with corresponding manual transcriptions (cf. 'legacy' distribution below).

- Dictionary with pronunciations (159848 entries), same file as the one included in TED-LIUM 2
- Selected monolingual data for language modeling from WMT12 publicly available corpora: these files come from the TED-LIUM 2 release, but have been modified to get a tokenization more relevant for English language

- Two corpus distributions:
-- the legacy one, on which the dev and test datasets are the same as in TED-LIUM 2 (and TED-LIUM 1).
-- the 'speaker adaptation' one, especially designed for experiments on speaker adaptation.

---

SPH format info:

Channels       : 1
Sample Rate    : 16000
Precision      : 16-bit
Bit Rate       : 256k
Sample Encoding: 16-bit Signed Integer PCM

---

François Hernandez, Vincent Nguyen, Sahar Ghannay, Natalia Tomashenko, and Yannick Estève, "TED-LIUM 3: twice as much data and corpus repartition for experiments on speaker adaptation", submitted to the 20th International Conference on Speech and Computer (SPECOM 2018), September 2018, Leipzig, Germany
A preprint version is available on arxiv (and in the doc/ directory):
https://arxiv.org/abs/1805.04699
"""
import logging
import shutil
import tarfile
from pathlib import Path
from typing import Dict, Optional, Union

from lhotse import (
    Recording,
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
    validate_recordings_and_supervisions,
)
from lhotse.utils import Pathlike, urlretrieve_progress


def download_tedlium(
    target_dir: Pathlike = ".", force_download: Optional[bool] = False
) -> Path:
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    tar_path = target_dir / "TEDLIUM_release-3.tgz"
    corpus_dir = target_dir / "TEDLIUM_release-3"
    completed_detector = corpus_dir / ".completed"
    if completed_detector.is_file():
        logging.info(f"Skipping {tar_path.name} because {completed_detector} exists.")
        return corpus_dir
    if force_download or not tar_path.is_file():
        urlretrieve_progress(
            "http://www.openslr.org/resources/51/TEDLIUM_release-3.tgz",
            filename=tar_path,
            desc="Downloading TEDLIUM v3",
        )
    shutil.rmtree(corpus_dir, ignore_errors=True)
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=target_dir)
    completed_detector.touch()
    return corpus_dir


def prepare_tedlium(
    tedlium_root: Pathlike, output_dir: Optional[Pathlike] = None
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Prepare manifests for the TED-LIUM v3 corpus.

    The manifests are created in a dict with three splits: train, dev and test.
    Each split contains a RecordingSet and SupervisionSet in a dict under keys 'recordings' and 'supervisions'.

    :param tedlium_root: Path to the unpacked TED-LIUM data.
    :return: A dict with standard corpus splits containing the manifests.
    """
    tedlium_root = Path(tedlium_root)
    output_dir = Path(output_dir) if output_dir is not None else None
    corpus = {}
    for split in ("train", "dev", "test"):
        root = tedlium_root / "legacy" / split
        recordings = RecordingSet.from_recordings(
            Recording.from_file(p) for p in (root / "sph").glob("*.sph")
        )
        stms = list((root / "stm").glob("*.stm"))
        assert len(stms) == len(recordings), (
            f"Mismatch: found {len(recordings)} "
            f"sphere files and {len(stms)} STM files. "
            f"You might be missing some parts of TEDLIUM..."
        )
        segments = []
        for p in stms:
            with p.open() as f:
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
                            text=text,
                            language="English",
                            speaker=rec_id,
                        )
                    )
        supervisions = SupervisionSet.from_segments(segments)
        corpus[split] = {"recordings": recordings, "supervisions": supervisions}

        validate_recordings_and_supervisions(**corpus[split])

        if output_dir is not None:
            recordings.to_json(output_dir / f"recordings_{split}.json")
            supervisions.to_json(output_dir / f"supervisions_{split}.json")

    return corpus
