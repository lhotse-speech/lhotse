"""
The Edinburgh International Accents of English Corpus

Citation

Sanabria, Ramon; Markl, Nina; Carmantini, Andrea; Klejch, Ondrej; Bell, Peter; Bogoychev, Nikolay. (2023). The Edinburgh International Accents of English Corpus, [dataset]. University of Edinburgh. School of Informatics. The Institute for Language, Cognition and Computation. The Centre for Speech Technology Research. https://doi.org/10.7488/ds/3832.

Description

English is the most widely spoken language in the world, used daily by millions of people as a first or second language in many different contexts.
As a result, there are many varieties of English.
Although the great many advances in English automatic speech recognition (ASR) over the past decades, results are usually reported based on test datasets which fail to represent the diversity of English as spoken today around the globe.
We present the first release of The Edinburgh International Accents of English Corpus (EdAcc).
This dataset attempts to better represent the wide diversity of English, encompassing almost 40 hours of dyadic video call conversations between friends.
Unlike other datasets, EdAcc includes a wide range of first and second-language varieties of English and a linguistic background profile of each speaker.
Results on latest public, and commercial models show that EdAcc highlights shortcomings of current English ASR models.
The best performing model, trained on 680 thousand hours of transcribed data, obtains an average of 19.7% WER -- in contrast to the the 2.7% WER obtained when evaluated on US English clean read speech.
Across all models, we observe a drop in performance on Jamaican, Indonesian, Nigerian, and Kenyan English speakers.
Recordings, linguistic backgrounds, data statement, and evaluation scripts are released on our website under CC-BY-SA.

Source: https://datashare.ed.ac.uk/handle/10283/4836
"""
import logging
import math
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, Optional, Union

from lhotse import fix_manifests, validate_recordings_and_supervisions
from lhotse.audio import RecordingSet
from lhotse.supervision import SupervisionSet
from lhotse.utils import Pathlike, is_module_available, resumable_download, safe_extract

_EDACC_SAMPLING_RATE = 32000


def download_edacc(
    target_dir: Pathlike = ".",
    force_download: bool = False,
    base_url: str = "https://datashare.ed.ac.uk/download/",
) -> Path:
    """
    Download and extract the EDACC dataset.

    :param target_dir: Pathlike, the path of the dir to store the dataset.
    :param force_download: Bool, if True, download the data even if it exists.
    :param base_url: str, the url of the website used to fetch the archive from.
    :return: the path to downloaded and extracted directory with data.
    """
    archive_name = "DS_10283_4836.zip"

    target_dir = Path(target_dir)
    corpus_dir = target_dir / "edacc"
    target_dir.mkdir(parents=True, exist_ok=True)

    completed_detector = corpus_dir / ".completed"
    if completed_detector.is_file():
        logging.info(f"Skipping EDACC download because {completed_detector} exists.")
        return corpus_dir

    # Maybe-download the archive.
    archive_path = target_dir / archive_name
    resumable_download(
        f"{base_url}/{archive_name}",
        filename=archive_path,
        force_download=force_download,
    )

    # Remove partial unpacked files, if any, and unpack everything.
    shutil.rmtree(corpus_dir, ignore_errors=True)
    with zipfile.ZipFile(archive_path) as zip:
        zip.extractall(path=corpus_dir)
    tar_name = "edacc_v1.0.tar.gz"
    with tarfile.open(corpus_dir / tar_name) as tar:
        safe_extract(tar, corpus_dir)
    completed_detector.touch()

    return corpus_dir


def prepare_edacc(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions.

    :param corpus_dir: a path to the unzipped EDACC directory (has ``edacc_v1.0`` inside).
    :param output_dir: an optional path where to write the manifests.
    :return: a dict with structure ``{"dev|test": {"recordings|supervisions": <manifest>}}``
    """
    from lhotse.kaldi import load_kaldi_data_dir

    if not is_module_available("pandas"):
        raise ValueError("Please install pandas via 'pip install pandas'.")

    corpus_dir = Path(corpus_dir) / "edacc_v1.0"
    audio_dir = corpus_dir / "data"
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    manifests = {}

    # Read extra metadata
    spk2meta = parse_linguistic_background(corpus_dir / "linguistic_background.csv")

    # Create recordings manifest and prepare data to create wav.scp files later.
    recordings = RecordingSet.from_dir(audio_dir, "*.wav")
    for r in recordings:
        assert r.num_channels == 1, f"Unexpected multi-channel recording: {r}"
        assert r.sampling_rate == _EDACC_SAMPLING_RATE
    wav_scp = {r.id: f"{r.id} {r.sources[0].source}" for r in recordings}

    for split in ("dev", "test"):
        data_dir = corpus_dir / split

        # First, create wav.scp in Kaldi data dir, and then just import it.
        with open(data_dir / "segments") as f:
            split_rec_ids = set(l.split()[1] for l in f)
        with open(data_dir / "wav.scp", "w") as f:
            for rid, rstr in sorted(wav_scp.items()):
                if rid in split_rec_ids:
                    print(rstr, file=f)
        recordings, supervisions, _ = load_kaldi_data_dir(
            data_dir, sampling_rate=_EDACC_SAMPLING_RATE
        )

        # Add extra metadata available.
        with open(data_dir / "conv.list") as f:
            conv_rec_ids = set(map(str.strip, f))
        for s in supervisions:
            s.language = "English"
            s.is_conversational = s.recording_id in conv_rec_ids
            for key, val in spk2meta[s.speaker].items():
                setattr(s, key, val)

        # Fix, validate, and save manifests.
        recordings, supervisions = fix_manifests(recordings, supervisions)
        validate_recordings_and_supervisions(recordings, supervisions)
        manifests[split] = {"recordings": recordings, "supervisions": supervisions}
        if output_dir is not None:
            recordings.to_file(output_dir / f"edacc_recordings_{split}.jsonl.gz")
            supervisions.to_file(output_dir / f"edacc_supervisions_{split}.jsonl.gz")

    return manifests


def parse_linguistic_background(path: Pathlike) -> Dict:
    import pandas as pd

    df = pd.read_csv(path)
    df = df.rename(
        columns={
            "What is your gender?": "gender",
            "What’s your ethnic background? ": "ethnicity",
            "What is your higher level of education?": "education",
            "How would you describe your accent in English? (e.g. Italian, Glaswegian)": "accent",
            "Do you speak any second languages? separate them with commas  (e.g., Mandarin,Catalan,French )": "other_languages",
            "What’s your year of birth? (e.g., 1992)": "birth_year",
            "What year did you start learning English? (e.g., 1999)": "start_english_year",
        }
    )
    df["age"] = 2022 - df.birth_year
    df["years_speaking_english"] = 2022 - df.start_english_year

    def parse(key, val) -> Optional:
        if key == "years_speaking_english":
            if math.isnan(val):
                return None
            return int(val)
        if key == "other_languages":
            if isinstance(val, float) and math.isnan(val):
                return []
            return [v.strip() for v in val.split(",")]
        if isinstance(val, str):
            return val.strip()
        return val

    spk2meta = {
        row["PARTICIPANT_ID"]: {
            m: parse(m, row[m])
            for m in (
                "gender",
                "ethnicity",
                "education",
                "accent",
                "other_languages",
                "age",
                "years_speaking_english",
            )
        }
        for _, row in df.iterrows()
    }

    return spk2meta
