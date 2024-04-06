"""
About the Earnings 22 dataset:

    The Earnings 22 dataset ( also referred to as earnings22 ) is a 119-hour corpus
    of English-language earnings calls collected from global companies. The primary
    purpose is to serve as a benchmark for industrial and academic automatic speech
    recognition (ASR) models on real-world accented speech.

    This dataset has been submitted to Interspeech 2022. The paper describing our
    methods and results can be found on arXiv at https://arxiv.org/abs/2203.15591.

    @misc{https://doi.org/10.48550/arxiv.2203.15591,
    doi = {10.48550/ARXIV.2203.15591},
    url = {https://arxiv.org/abs/2203.15591},
    author = {Del Rio, Miguel and Ha, Peter and McNamara, Quinten and Miller, Corey and Chandra, Shipra},
    keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
    title = {Earnings-22: A Practical Benchmark for Accents in the Wild},
    publisher = {arXiv},
    year = {2022},
    copyright = {Creative Commons Attribution Share Alike 4.0 International}
    }

"""


import logging
import string
from pathlib import Path
from typing import Dict, List, Optional, Union

from lhotse import fix_manifests, validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike

_DEFAULT_URL = "https://github.com/revdotcom/speech-datasets"


def normalize(text: str) -> str:
    # Remove all punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Convert all upper case to lower case
    text = text.lower()
    return text


def read_metadata(path: Pathlike) -> Dict[str, List[str]]:
    with open(path) as f:
        f.readline()  # skip header
        out = dict()
        for line in f:
            line = line.split(",")
            out[line[0]] = line[1:-1]
        return out


def download_earnings22(
    target_dir: Pathlike = ".",
    force_download: Optional[bool] = False,
    url: Optional[str] = _DEFAULT_URL,
) -> Path:
    """Download and untar the dataset.
    :param target_dir: Pathlike, the path of the dir to store the dataset.
        The extracted files are saved to target_dir/earnings22/
        Please note that the github repository contains other additional datasets and
        using this call, you will be downloading all of them and then throwing them out.
    :param force_download: Bool, if True, download the tar file no matter
        whether it exists or not.
    :param url: str, the url to download the dataset.
    :return: the path to downloaded and extracted directory with data.
    """
    logging.error(
        "Downloading Earnings22 from github repository is not implemented. "
        + f"Please visit {_DEFAULT_URL} and download the files manually. Please "
        + "follow the instructions closely as you need to use git-lfs to download "
        + "some of the audio files."
    )


def parse_nlp_file(filename: Pathlike):
    with open(filename) as f:
        transcript = list()
        f.readline()  # skip header
        for line in f:
            line = line.split("|")
            transcript.append(line[0])
        return transcript


def prepare_earnings22(
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
    assert len(audio_files) == 125

    audio_files.sort()
    recording_set = RecordingSet.from_recordings(
        Recording.from_file(p) for p in audio_files
    )

    nlp_dir = corpus_dir / "transcripts" / "nlp_references"
    nlp_files = list(nlp_dir.glob("*.nlp"))
    assert len(nlp_files) == 125

    metadata = read_metadata(corpus_dir / "metadata.csv")

    nlp_files.sort()
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
            language=f"English-{metadata[id][4]}",
            text=text,
        )
        supervision_segments.append(s)
    supervision_set = SupervisionSet.from_segments(supervision_segments)

    recording_set, supervision_set = fix_manifests(recording_set, supervision_set)
    validate_recordings_and_supervisions(recording_set, supervision_set)
    if output_dir is not None:
        supervision_set.to_file(output_dir / "earnings22_supervisions_all.jsonl.gz")
        recording_set.to_file(output_dir / "earnings22_recordings_all.jsonl.gz")

    return recording_set, supervision_set
