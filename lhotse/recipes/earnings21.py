"""
About the Earnings 21 dataset:

    The Earnings 21 dataset ( also referred to as earnings21 ) is a 39-hour corpus of
    earnings calls containing entity dense speech from nine different financial sectors.
    This corpus is intended to benchmark automatic speech recognition (ASR) systems
    in the wild with special attention towards named entity recognition (NER).

    This dataset has been submitted to Interspeech 2021. The paper describing methods
    and results can be found on arXiv at https://arxiv.org/pdf/2104.11348.pdf

    @misc{delrio2021earnings21,
        title={Earnings-21: A Practical Benchmark for ASR in the Wild},
        author={Miguel Del Rio and Natalie Delworth and Ryan Westerman and Michelle Huang and Nishchal Bhandari and Joseph Palakapilly and Quinten McNamara and Joshua Dong and Piotr Zelasko and Miguel Jette},
        year={2021},
        eprint={2104.11348},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
    }

"""


import logging
import shutil
import string
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from lhotse import fix_manifests, validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, resumable_download

_DEFAULT_URL = (
    "https://codeload.github.com/revdotcom/speech-datasets/zip/refs/heads/main"
)


def normalize(text: str) -> str:
    # Remove all punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Convert all upper case to lower case
    text = text.lower()
    return text


def download_earnings21(
    target_dir: Pathlike = ".",
    force_download: Optional[bool] = False,
    url: Optional[str] = _DEFAULT_URL,
) -> Path:
    """Download and untar the dataset.
    :param target_dir: Pathlike, the path of the dir to store the dataset.
        The extracted files are saved to target_dir/earnings21/
        Please note that the github repository contains other additional datasets and
        using this call, you will be downloading all of them and then throwing them out.
    :param force_download: Bool, if True, download the tar file no matter
        whether it exists or not.
    :param url: str, the url to download the dataset.
    :return: the path to downloaded and extracted directory with data.
    """
    logging.info(
        "Downloading Earnings21 from github repository is not very efficient way"
        + " how to obtain the corpus. You will be downloading other data as well."
    )
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir = target_dir / "earnings21"

    zip_path = target_dir / "speech-datasets-main.zip"

    completed_detector = extracted_dir / ".lhotse-download.completed"
    if completed_detector.is_file():
        logging.info(f"Skipping - {completed_detector} exists.")
        return extracted_dir
    resumable_download(url, filename=zip_path, force_download=force_download)
    shutil.rmtree(extracted_dir, ignore_errors=True)
    with zipfile.ZipFile(zip_path) as zip:
        for f in zip.namelist():
            if "earnings21" in f:
                zip.extract(f, path=target_dir)

    # For Python < 3.9, shutil.move() gives error with PosixPath
    shutil.move(
        str(target_dir / "speech-datasets-main" / "earnings21"), str(target_dir)
    )
    shutil.rmtree(target_dir / "speech-datasets-main")

    completed_detector.touch()

    return extracted_dir


def parse_nlp_file(filename: Pathlike):
    with open(filename) as f:
        transcript = list()
        f.readline()  # skip header
        for line in f:
            line = line.split("|")
            transcript.append(line[0])
        return transcript


def prepare_earnings21(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    normalize_text: bool = False,
) -> Union[RecordingSet, SupervisionSet]:
    """
    Returns the manifests which consist of the Recordings and Supervisions.
    When all the manifests are available in the ``output_dir``, it will simply
    read and return them.

    :param corpus_dir: Pathlike, the path of the data dir. The structure is
        expected to mimic the structure in the github repository, notably
        the mp3 files will be searched for in [corpus_dir]/media and transcriptions
        in the directory [corpus_dir]/transcripts/nlp_references
    :param output_dir: Pathlike, the path where to write the manifests.
    :param normalize_text: Bool, if True, normalize the text.
    :return: (recordings, supervisions) pair

    .. caution::
        The `normalize_text` option removes all punctuation and converts all upper case
        to lower case. This includes removing possibly important punctuations such as
        dashes and apostrophes.
    """

    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    media_dir = corpus_dir / "media"
    audio_files = list(media_dir.glob("*.mp3"))
    assert len(audio_files) == 44

    audio_files.sort()
    recording_set = RecordingSet.from_recordings(
        Recording.from_file(p) for p in audio_files
    )

    nlp_dir = corpus_dir / "transcripts" / "nlp_references"
    nlp_files = list(nlp_dir.glob("*.nlp"))
    assert len(nlp_files) == 44

    supervision_segments = list()
    for nlp_file in nlp_files:
        id = nlp_file.stem
        text = " ".join(parse_nlp_file(nlp_file))
        if normalize_text:
            text = normalize(text)

        s = SupervisionSegment(
            id=id,
            recording_id=id,
            start=0.0,
            duration=recording_set[id].duration,  # recording.duration,
            channel=0,
            language="English",
            text=text,
        )
        supervision_segments.append(s)
    supervision_set = SupervisionSet.from_segments(supervision_segments)

    recording_set, supervision_set = fix_manifests(recording_set, supervision_set)
    validate_recordings_and_supervisions(recording_set, supervision_set)
    if output_dir is not None:
        supervision_set.to_file(output_dir / "earnings21_supervisions_all.jsonl.gz")
        recording_set.to_file(output_dir / "earnings21_recordings_all.jsonl.gz")

    return recording_set, supervision_set
