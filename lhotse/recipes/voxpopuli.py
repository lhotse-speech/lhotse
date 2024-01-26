"""
VoxPopuli provides

- 400K hours of unlabelled speech data for 23 languages
- 1.8K hours of transcribed speech data for 16 languages
- 17.3K hours of speech-to-speech interpretation data for 15x15 directions
- 29 hours of transcribed speech data of non-native English intended for research in ASR
for accented speech (15 L2 accents)

The raw data is collected from 2009-2020 European Parliament event recordings.
For details about the corpus, please refer to the website:
https://github.com/facebookresearch/voxpopuli

Reference:
Wang, Changhan et al. “VoxPopuli: A Large-Scale Multilingual Speech Corpus for Representation
Learning, Semi-Supervised Learning and Interpretation.” Annual Meeting of the Association
for Computational Linguistics (2021).

This script is based on code from the repository linked above.

NOTE: Our data preparation is slightly different from the original repository. In particular,
we only use the metadata to create manifests, i.e., we do not create segment-level wav files,
unlike the original repository. In this way, we can avoid duplicating the audio files.
"""
import csv
import gzip
import logging
import re
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from torch.hub import download_url_to_file
from tqdm import tqdm

from lhotse import RecordingSet, SupervisionSegment, SupervisionSet
from lhotse.qa import fix_manifests, validate_recordings_and_supervisions
from lhotse.utils import Pathlike, safe_extract

# fmt: off
LANGUAGES = [
    "en", "de", "fr", "es", "pl", "it", "ro", "hu", "cs", "nl", "fi", "hr",
    "sk", "sl", "et", "lt", "pt", "bg", "el", "lv", "mt", "sv", "da"
]
LANGUAGES_V2 = [f"{x}_v2" for x in LANGUAGES]

YEARS = list(range(2009, 2020 + 1))

ASR_LANGUAGES = [
    "en", "de", "fr", "es", "pl", "it", "ro", "hu", "cs", "nl", "fi", "hr",
    "sk", "sl", "et", "lt"
]
ASR_ACCENTED_LANGUAGES = [
    "en_accented"
]

S2S_SRC_LANGUAGES = ASR_LANGUAGES

S2S_TGT_LANGUAGES = [
    "en", "de", "fr", "es", "pl", "it", "ro", "hu", "cs", "nl", "fi", "hr",
    "sk", "sl", "et", "lt", "pt", "bg", "el", "lv", "mt", "sv", "da"
]

S2S_TGT_LANGUAGES_WITH_HUMAN_TRANSCRIPTION = ["en", "fr", "es"]

DOWNLOAD_BASE_URL = "https://dl.fbaipublicfiles.com/voxpopuli"
# fmt: on


def download_voxpopuli(
    target_dir: Pathlike = ".",
    subset: Optional[str] = "asr",
) -> Path:
    """
    Download and untar/unzip the VoxPopuli dataset.

    :param target_dir: Pathlike, the path of the dir to storage the dataset.
    :param subset: str, the subset of the dataset to download, can be one of "400k", "100k",
        "10k", "asr", or any of the languages in LANGUAGES or LANGUAGES_V2.
    :return: the path to downloaded and extracted directory with data.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if subset in LANGUAGES_V2:
        languages = [subset.split("_")[0]]
        years = YEARS + [f"{y}_2" for y in YEARS]
    elif subset in LANGUAGES:
        languages = [subset]
        years = YEARS
    else:
        languages = {
            "400k": LANGUAGES,
            "100k": LANGUAGES,
            "10k": LANGUAGES,
            "asr": ["original"],
        }.get(subset, None)
        years = {
            "400k": YEARS + [f"{y}_2" for y in YEARS],
            "100k": YEARS,
            "10k": [2019, 2020],
            "asr": YEARS,
        }.get(subset, None)

    url_list = []
    for l in languages:
        for y in years:
            url_list.append(f"{DOWNLOAD_BASE_URL}/audios/{l}_{y}.tar")

    out_root = target_dir / "raw_audios"
    out_root.mkdir(exist_ok=True, parents=True)
    logging.info(f"{len(url_list)} files to download...")
    for url in tqdm(url_list):
        tar_path = out_root / Path(url).name
        download_url_to_file(url, tar_path)
        with tarfile.open(tar_path, "r") as tar_file:
            safe_extract(tar_file, out_root)
        tar_path.unlink()

    return target_dir


def prepare_voxpopuli(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    task: str = "asr",
    lang: str = "en",
    source_lang: Optional[str] = None,
    target_lang: Optional[str] = None,
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Prepares and returns the VoxPopuli manifests which consist of Recordings and Supervisions.

    :param corpus_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param task: str, the task to prepare the manifests for, can be one of "asr", "s2s", "lm".
    :param lang: str, the language to prepare the manifests for, can be one of LANGUAGES
        or LANGUAGES_V2. This is used for "asr" and "lm" tasks.
    :param source_lang: str, the source language for the s2s task, can be one of S2S_SRC_LANGUAGES.
    :param target_lang: str, the target language for the s2s task, can be one of S2S_TGT_LANGUAGES.
    :param num_jobs: int, the number of parallel jobs to use for preparing the manifests.
    :return: Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]], the manifests.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

    if task == "asr":
        assert lang in ASR_LANGUAGES, f"Unsupported language: {lang}"
        manifests = _prepare_voxpopuli_asr(
            corpus_dir, output_dir, lang, num_jobs=num_jobs
        )
    elif task == "s2s":
        assert (
            source_lang in S2S_SRC_LANGUAGES
        ), f"Unsupported source language: {source_lang}"
        assert (
            target_lang in S2S_TGT_LANGUAGES
        ), f"Unsupported target language: {target_lang}"
        manifests = _prepare_voxpopuli_s2s(corpus_dir, source_lang, target_lang)
    elif task == "lm":
        assert lang in ASR_LANGUAGES, f"Unsupported language: {lang}"
        manifests = _prepare_voxpopuli_lm(corpus_dir, lang)

    for k, v in manifests.items():
        recordings, supervisions = fix_manifests(**v)
        validate_recordings_and_supervisions(
            recordings=recordings, supervisions=supervisions
        )
        manifests[k]["recordings"] = recordings
        manifests[k]["supervisions"] = supervisions

        lang_affix = f"{source_lang}-{target_lang}" if task == "s2s" else lang
        if output_dir is not None:
            recordings.to_file(
                output_dir / f"voxpopuli-{task}-{lang_affix}_recordings_{k}.jsonl.gz"
            )
            supervisions.to_file(
                output_dir / f"voxpopuli-{task}-{lang_affix}_supervisions_{k}.jsonl.gz"
            )

    return manifests


class RecordingIdFn:
    """
    This functor class avoids error in multiprocessing:
    `AttributeError: Can't pickle local object '_prepare_voxpopuli_asr.<locals>.<lambda>'`
    """

    def __init__(self, language: str):
        self.language = language

    def __call__(self, path: Path) -> str:
        recording_id = re.sub(f"_{self.language}$", "", path.stem)
        recording_id = re.sub("_original$", "", recording_id)
        return recording_id


def _prepare_voxpopuli_asr(
    corpus_dir: Path, output_dir: Path, lang: str, num_jobs: int = 1
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Download metadata TSV and prepare manifests for the ASR task.
    """
    # First create recordings. We remove the affix "_original" from the recording ID
    logging.info("Preparing recordings (this may take a few minutes)...")
    in_root = corpus_dir / "raw_audios" / lang
    recordings = RecordingSet.from_dir(
        in_root,
        "*.ogg",
        num_jobs=num_jobs,
        recording_id=RecordingIdFn(language=lang),
    )

    # Now create supervisions

    # Get metadata TSV
    url = f"{DOWNLOAD_BASE_URL}/annotations/asr/asr_{lang}.tsv.gz"

    tsv_path = output_dir / Path(url).name

    if not tsv_path.exists():
        logging.info(f"Downloading : {url} -> {tsv_path}")
        download_url_to_file(url, tsv_path)
    else:
        logging.info(f"Using pre-downloaded annotations {tsv_path}")

    with gzip.open(tsv_path, "rt") as f:
        metadata = [x for x in csv.DictReader(f, delimiter="|")]

    # Get segment into list (train, dev, test)
    segments = defaultdict(list)
    # We also keep a count of the number of segments per recording
    num_segments = defaultdict(lambda: 0)

    for r in tqdm(metadata):
        split = r["split"]
        if split not in ["train", "dev", "test"]:
            continue
        reco_id = r["session_id"]
        start_time = float(r["start_time"])
        duration = float(r["end_time"]) - start_time

        num_segments[reco_id] += 1
        segments[split].append(
            SupervisionSegment(
                id=f"{reco_id}-{num_segments[reco_id]}",
                recording_id=reco_id,
                start=round(start_time, ndigits=8),
                duration=round(duration, ndigits=8),
                channel=0,
                language=lang,
                speaker=r["speaker_id"],
                gender=r["gender"],
                text=r["normed_text"],
                custom={
                    "orig_text": r["original_text"],
                },
            )
        )

    # Get list of recording IDs for each split
    reco_ids = defaultdict(list)
    for split, segs in segments.items():
        reco_ids[split] = sorted(set([s.recording_id for s in segs]))

    manifests = defaultdict(dict)
    for split in ["train", "dev", "test"]:
        manifests[split]["recordings"] = recordings.filter(
            lambda r: r.id in reco_ids[split]
        )
        manifests[split]["supervisions"] = SupervisionSet.from_segments(segments[split])

    return manifests


def _prepare_voxpopuli_s2s(
    corpus_dir: Path, source_lang: str, target_lang: str
) -> Tuple[RecordingSet, SupervisionSet]:
    """
    Prepare the manifests for the s2s task.
    """
    raise NotImplementedError


def _prepare_voxpopuli_lm(corpus_dir: Path, lang: str) -> Tuple[RecordingSet, None]:
    """
    Prepare the manifests for the lm task.
    """
    raise NotImplementedError
