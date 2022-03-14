"""
MTEDx is a collection of transcribed and translated speech corpora:
 https://openslr.org/100

It has 8 languages:
es - 189h
fr - 189h
pt - 164h
it - 107h
ru - 53h
el - 30h
ar - 18h
de - 15h

A subset of this audio is translated and split into the following partitions:
     - train
     - dev
     - test
     - iwslt2021 sets

This recipe only prepares the ASR portion of the data.
"""
import logging
import re
import tarfile
import unicodedata
from collections import defaultdict
from concurrent.futures.thread import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

from tqdm.auto import tqdm

from lhotse import (
    Recording,
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
    validate_recordings_and_supervisions,
)
from lhotse.qa import (
    remove_missing_recordings_and_supervisions,
    trim_supervisions_to_recordings,
)
from lhotse.utils import Pathlike, is_module_available, urlretrieve_progress

# Keep Markings such as vowel signs, all letters, and decimal numbers
VALID_CATEGORIES = ("Mc", "Mn", "Ll", "Lm", "Lo", "Lt", "Lu", "Nd", "Zs")
KEEP_LIST = ["\u2019"]


ASR = (
    "es",
    "fr",
    "pt",
    "it",
    "ru",
    "el",
    "ar",
    "de",
)


ISOCODE2LANG = {
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "it": "Italian",
    "ru": "Russian",
    "el": "Greek",
    "ar": "Arabic",
    "de": "German",
}

###############################################################################
#                             Download and Untar
###############################################################################
def download_mtedx(
    target_dir: Pathlike = ".",
    languages: Optional[Union[str, Sequence[str]]] = "all",
) -> Path:
    """
    Download and untar the dataset.

    :param: target_dir: Pathlike, the path of the directory where the
        mtdex_corpus directory will be created and to which data will
        be downloaded.
    :param: languages: A str or sequence of strings specifying which
        languages to download. The default 'all', downloads all available
        languages.
    :return: the path to downloaded and extracted directory with data.
    """
    target_dir = Path(target_dir) / "mtedx_corpus"
    target_dir.mkdir(parents=True, exist_ok=True)

    langs_list = list(ISOCODE2LANG.keys())
    # If for some reason languages = None, assume this also means 'all'
    if isinstance(languages, str) and languages != "all":
        langs_list = [languages]
    elif isinstance(languages, list) or isinstance(languages, tuple):
        langs_list = languages

    for lang in tqdm(langs_list, "Downloading MTEDx languages"):
        tar_path = target_dir / f"{lang}-{lang}.tgz"
        completed_detector = target_dir / f".{lang}.completed"
        if completed_detector.is_file():
            logging.info(f"Skipping {lang} because {completed_detector} exists.")
            continue
        urlretrieve_progress(
            f"http://www.openslr.org/resources/100/mtedx_{lang}.tgz",
            filename=tar_path,
            desc=f"Downloading MTEDx {lang}",
        )
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=target_dir)
        completed_detector.touch()

    return target_dir


###############################################################################
#                              Prepare MTEDx
###############################################################################
def prepare_mtedx(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    languages: Optional[Union[str, Sequence[str]]] = "all",
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]]:
    """
    Prepares manifest of all MTEDx languages requested.

    :param corpus_dir: Path to the root where MTEDx data was downloaded.
                       It should be called mtedx_corpus.
    :param output_dir: Root directory where .json manifests are stored.
    :param languages: str or str sequence specifying the languages to prepare.
        The str 'all' prepares all languages.
    :return:
    """
    # Resolve corpus_dir type
    if isinstance(corpus_dir, str):
        corpus_dir = Path(corpus_dir)

    # Resolve output_dir type
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    langs_list = list(ISOCODE2LANG.keys())
    # If for some reason languages = None, assume this also means 'all'
    if isinstance(languages, str) and languages != "all":
        langs_list = [languages]
    elif isinstance(languages, list) or isinstance(languages, tuple):
        if languages[0] != "all":
            langs_list = languages

    manifests = defaultdict(dict)
    for lang in langs_list:
        corpus_dir_lang = corpus_dir / f"{lang}-{lang}"
        output_dir_lang = output_dir / f"{lang}"
        if corpus_dir_lang.is_dir():
            manifests[lang] = prepare_single_mtedx_language(
                corpus_dir_lang,
                output_dir_lang,
                language=lang,
                num_jobs=num_jobs,
            )

    return dict(manifests)


###############################################################################
# All remaining functions are just helper functions, mainly for text
# normalization and parsing the vtt files that come with the mtedx corpus
###############################################################################

# Prepare data for a single language
def prepare_single_mtedx_language(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    language: str = "language",
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Prepares manifests using a single MTEDx language.

    This function works as follows:

        - First it looks for the audio directory in the data/wav where the .flac
            files are stored.
        - Then, it looks for the vtt directory in data/{train,dev,test}/vtt
            which contains the segmentation and transcripts for the audio.
        - The transcripts undergo some basic text normalization

    :param corpus_dir: Path to the root of the MTEDx download
    :param output_dir: Path where the manifests are stored as .json files
    :param language: The two-letter language code.
    :param num_jobs: Number of threads to use when preparing data.
    :return:
    """
    if isinstance(corpus_dir, str):
        corpus_dir = Path(corpus_dir)
    manifests = defaultdict(dict)

    with ThreadPoolExecutor(num_jobs) as ex:
        for split in ("train", "valid", "test"):
            audio_dir = corpus_dir / f"data/{split}/wav"
            recordings = RecordingSet.from_recordings(
                Recording.from_file(p) for p in audio_dir.glob("*.flac")
            )
            if len(recordings) == 0:
                logging.warning(f"No .flac files found in {audio_dir}")

            supervisions = []
            text_dir = corpus_dir / f"data/{split}/vtt"
            futures = []
            for p in text_dir.glob("*"):
                futures.append(ex.submit(_filename_to_supervisions, p, language))

            for future in tqdm(futures, desc="Processing", leave=False):
                result = future.result()
                if result is None:
                    continue
                for sup in result:
                    supervisions.append(sup)

            if len(supervisions) == 0:
                logging.warning(f"No supervisions found in {text_dir}")
            supervisions = SupervisionSet.from_segments(supervisions)

            recordings, supervisions = remove_missing_recordings_and_supervisions(
                recordings, supervisions
            )
            supervisions = trim_supervisions_to_recordings(recordings, supervisions)
            validate_recordings_and_supervisions(recordings, supervisions)

            manifests[split] = {
                "recordings": recordings,
                "supervisions": supervisions,
            }

            if output_dir is not None:
                if isinstance(output_dir, str):
                    output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                save_split = "dev" if split == "valid" else split
                recordings.to_file(output_dir / f"recordings_{language}_{split}.json")
                supervisions.to_file(
                    output_dir / f"supervisions_{language}_{split}.json"
                )

    return dict(manifests)


# Function that takes in the vtt file name (Path object) and returns a
# list of SUpervisionSegments
def _filename_to_supervisions(filename: Path, language: str):
    lines = filename.read_text()
    recoid = filename.stem.split(".")[0]
    supervisions = []
    filterfun = partial(_filter_word)
    for start, end, line in _parse_vtt(lines, "<noise>"):
        line_list = []
        for w in line.split():
            w_ = w.strip()
            if re.match(r"^(\([^)]*\) *)+$", w_):
                line_list.append(w_)
            elif filterfun(w):
                line_list.append(w_)
            else:
                line_list.append("<unk>")
        line_ = " ".join(line_list)
        if re.match(r"^\w+ *(<[^>]*> *)+$", line_, re.UNICODE):
            line_new = line_.strip()
        elif "<" in line_ or ">" in line_:
            continue
        else:
            line_new = line_.strip()

        supervisions.append(
            SupervisionSegment(
                id=_format_uttid(recoid, start),
                recording_id=recoid,
                start=start,
                duration=round(end - start, ndigits=8),
                channel=0,
                text=line_new,
                language=language,
                speaker=recoid,
            )
        )
    return supervisions


# Handles proper formatting (0-padding) of utterance ids
def _format_uttid(recoid, start):
    # Since each recording is a talk, normally by 1 speaker, we use the
    # recoid as the spkid as well.
    start = "{0:08d}".format(int(float(start) * 100))
    return "_".join([recoid, start])


# Filters out words from text that do not have the right unicode categories.
# This includes strange punctuation for instance
def _filter_word(s):
    for c in s:
        if unicodedata.category(c) not in VALID_CATEGORIES and c not in KEEP_LIST:
            return False
    return True


# This filters out individual characters in a line of text
def _filter(s):
    return unicodedata.category(s) in VALID_CATEGORIES or s in KEEP_LIST


# Convert time to seconds
def _time2sec(time):
    hr, mn, sec = time.split(":")
    return int(hr) * 3600.0 + int(mn) * 60.0 + float(sec)


# Parse the times in the vtt files
def _parse_time_segment(l):
    start, end = l.split(" --> ")
    start = _time2sec(start)
    end = _time2sec(end)
    return start, end


# There are some strange space symbols that we normalize
def _normalize_space(c):
    if unicodedata.category(c) == "Zs":
        return " "
    else:
        return c


# Parse the vtt file
def _parse_vtt(lines, noise):
    # Import regex for some special unicode handling that re has issues with
    if not is_module_available("regex"):
        raise ImportError(
            "regex package not found. Please install..." " (pip install regex)"
        )
    else:
        import regex as re2

    noise_pattern = re2.compile(r"\([^)]*\)", re2.UNICODE)
    apostrophe_pattern = re2.compile(r"(\w)'(\w)")
    html_tags = re2.compile(r"(&[^ ;]*;)|(</?[iu]>)")

    blocks = lines.split("\n\n")
    for i, b in enumerate(blocks, -1):
        if i > 0 and b.strip() != "":
            b_lines = b.split("\n")
            start, end = _parse_time_segment(b_lines[0])
            line = " ".join(b_lines[1:])
            line_new = line
            if line.strip("- ") != "":
                line_parts = noise_pattern.sub(noise, line_new)
                line_parts = apostrophe_pattern.sub(r"\1\u2019\2", line_parts)
                line_parts = html_tags.sub("", line_parts)
                line_parts_new = []
                for lp in line_parts.split(noise):
                    line_parts_new.append(
                        "".join(
                            [i for i in filter(_filter, lp.strip().replace("-", " "))]
                        )
                    )
                joiner = " " + noise + " "
                line_new = joiner.join(line_parts_new)
                line_new = re2.sub(
                    r"\p{Zs}", lambda m: _normalize_space(m.group(0)), line_new
                )
                line_new = re2.sub(r" +", " ", line_new).strip().lower()
            yield start, end, line_new
