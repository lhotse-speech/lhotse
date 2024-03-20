"""
About the Fisher Spanish corpus

    This is conversational telephone speech collected as 2-channel μ-law, 8kHz-sampled data. 
    The catalog number LDC2010S01 for audio corpus and LDC2010T04 for transcripts.

    This data is not available for free - your institution needs to have an LDC subscription.
    You also should download and prepare the pre-defined splits with:
        git clone https://github.com/joshua-decoder/fisher-callhome-corpus.git
        cd fisher-callhome-corpus
        make
        cd ../
"""

from bs4 import BeautifulSoup
import codecs
import itertools as it
import math
import os
import logging
import re
import string
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from tqdm.auto import tqdm

from lhotse.audio import RecordingSet
from lhotse.qa import fix_manifests, validate_recordings_and_supervisions
from lhotse.recipes.fisher_english import create_recording
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, check_and_rglob


def create_supervision(
    sessions: Dict[str, Dict[str, str]],
    split_map: Dict[Tuple[str, str], Tuple[str, str, List[int]]],
    splits_dir: Pathlike,
    trans_path: Pathlike,
    split: str,
    remove_punc: bool,
    lowercase: bool,
) -> List[SupervisionSegment]:

    trans_path = trans_path / "data" / "transcripts"
    splits_path = splits_dir / "mapping" / split
    # load info: duration, translation, channel
    tdf_files = {}
    idx = 0
    mapping_file = open(splits_path, "r").readlines()
    segments = []
    with tqdm(total=len(mapping_file), desc=f"Create {split} supervisions") as pbar:
        for l in mapping_file:
            file_name, _ = l.strip().split()
            if file_name not in tdf_files:
                tdf_files[file_name] = [
                    line.strip()
                    for line in open(trans_path / f"{file_name}.tdf", "r").readlines()
                ][3:]
            file = tdf_files[file_name]
            text_es, text_en, line_num = split_map[(file_name, idx)]
            text_es = fix_small_errors(text_es)
            text_es, lids, cs = process_cs(text_es, remove_punc, lowercase)

            text_en = clean(text_en, remove_punc, lowercase)
            text_es = clean(text_es, remove_punc, lowercase)

            s_time = round(float(file[line_num[0] - 1].split("\t")[2]), 10)
            e_time = round(float(file[line_num[-1] - 1].split("\t")[3]), 10)

            # fix mistakes with some channels
            if file_name == "20050930_180411_157_fsp" and line_num[0] - 1 == 1:
                ch = 0
            elif file_name == "20051225_192042_872_fsp" and line_num[0] - 1 == 0:
                ch = 0
            else:
                ch = int(file[line_num[0] - 1].split("\t")[1])
            segments.append(
                SupervisionSegment(
                    id=file_name + "-" + str(idx).zfill(len(str(len(split_map)))),
                    recording_id=file_name,
                    start=s_time,
                    duration=round(e_time - s_time, 10),
                    channel=ch,
                    text=text_es,
                    language=lids,
                    speaker=sessions[file_name.split("_")[2]][ch],
                    custom={"translated_text": {"en": text_en}, "cs": cs},
                )
            )
            idx += 1
            pbar.update()
    return segments


def prepare_fisher_spanish(
    audio_dir_path: Pathlike,
    transcript_dir_path: Pathlike,
    split_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    absolute_paths: bool = True,
    remove_punc: bool = False,
    lowercase: bool = False,
    cs_splits: bool = True,
    num_jobs: int = 4,
) -> Dict[str, Union[RecordingSet, SupervisionSet]]:
    """
    Prepares manifests for Fisher Spanish with following:
        1\ mono and CS ASR 2\ Sp-En ST
    :param audio_dir_path: Path to audio directory (usually LDC2010S01).
    :param transcript_dir_path: Path to transcript directory (usually LDC2010T04).
    :param split_dir: Path to splits from https://github.com/joshua-decoder/fisher-callhome-corpus.git
    :param output_dir: Directory where the manifests should be written. Can be omitted to avoid writing.
    :param absolute_paths: Whether to return absolute or relative (to the corpus dir) paths for recordings.
    :param remove_punc: Remove punctuations from the text
    :param lowercase: Lower case the text
    :param cs_splits: Whether to create CS splits
    :return: A dict with manifests. The keys are: ``{'recordings', 'supervisions'}``.
    """

    manifests = {}
    split_map = load_splits(split_dir)
    audio_dir_path, transcript_dir_path = Path(audio_dir_path), Path(
        transcript_dir_path
    )

    audio_paths = check_and_rglob(audio_dir_path, "*.sph")
    transcript_paths = check_and_rglob(transcript_dir_path, "*.tdf")

    sessions_data_path = check_and_rglob(transcript_dir_path, "*_call.tbl")[0]
    with codecs.open(sessions_data_path, "r", "utf8") as sessions_data_f:
        session_lines = [
            l.rstrip("\n").split(",") for l in sessions_data_f.readlines()
        ][1:]
        sessions = {l[0]: {0: l[2], 1: l[8]} for l in session_lines}

    assert len(transcript_paths) == len(sessions) == len(audio_paths)

    create_recordings_input = [(p, None if absolute_paths else 4) for p in audio_paths]
    recordings = [None] * len(audio_paths)
    with ThreadPoolExecutor(os.cpu_count() * num_jobs) as executor:
        with tqdm(total=len(audio_paths), desc="Collect recordings") as pbar:
            for i, reco in enumerate(
                executor.map(create_recording, create_recordings_input)
            ):
                recordings[i] = reco
                pbar.update()
    recordings = RecordingSet.from_recordings(recordings)

    # prepare supervisions for each split
    for split in split_map:
        logging.info(f"Creating {split} supervisions")
        supervisions = create_supervision(
            sessions,
            split_map[split],
            split_dir,
            transcript_dir_path,
            split,
            remove_punc,
            lowercase,
        )
        supervisions = SupervisionSet.from_segments(supervisions).filter(
            lambda s: s.duration > 0.0
        )
        recordings_tmp, supervisions = fix_manifests(recordings, supervisions)
        validate_recordings_and_supervisions(recordings_tmp, supervisions)

        split_set = split.split("_")[1]
        if output_dir is not None:
            logging.info(f"Saving {split_set}")
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            recordings_tmp.to_file(
                output_dir / f"fisher-spanish_recordings_{split_set}.jsonl.gz"
            )
            supervisions.to_file(
                output_dir / f"fisher-spanish_supervisions_{split_set}.jsonl.gz"
            )
        if cs_splits:
            logging.info(f"Extracting and saving CS {split_set}")
            cs_sup = supervisions.filter(lambda s: s.custom["cs"] == 1)
            recordings_tmp, cs_sup = fix_manifests(recordings_tmp, cs_sup)
            validate_recordings_and_supervisions(recordings_tmp, cs_sup)
            recordings_tmp.to_file(
                output_dir / f"fisher-spanish_cs_recordings_{split_set}.jsonl.gz"
            )
            cs_sup.to_file(
                output_dir / f"fisher-spanish_cs_supervisions_{split_set}.jsonl.gz"
            )

        manifests[split] = {"recordings": recordings_tmp, "supervisions": supervisions}
    return manifests


def load_splits(splits_dir):
    splits = {}
    for split in ("fisher_train", "fisher_dev", "fisher_dev2", "fisher_test"):
        mapping = {}
        splits_path = splits_dir / "mapping" / split

        # load transcriptions (text_es) and tranlations (text_en)
        transcription_file = splits_dir / "corpus" / "ldc" / f"{split}.es"
        if "train" not in split:
            # *en.0 is usually chosen in other recipies
            translation_file = splits_dir / "corpus" / "ldc" / f"{split}.en.0"
        else:
            translation_file = splits_dir / "corpus" / "ldc" / f"{split}.en"
        transcriptions = [
            line.replace("\r", " ").strip()
            for line in open(transcription_file, "r", newline="\n").readlines()
        ]
        translations = [
            line.replace("\r", " ").strip()
            for line in open(translation_file, "r", newline="\n").readlines()
        ]
        assert len(transcriptions) == len(
            translations
        ), "Make sure lenght of fisher_train.en and fisher_train.es is the same"

        # load mapping
        idx = 0
        mapping_file = open(splits_path, "r").readlines()
        for l in mapping_file:
            file_name, line_num = l.strip().split()
            line_num = [int(i) for i in line_num.split("_")]
            mapping[(file_name, idx)] = (
                transcriptions[idx],
                translations[idx],
                line_num,
            )
            idx += 1
        splits[split] = mapping
    return splits


def process_cs(line, rm_punc=False, lc=False):
    """Function to detect if sentence contains English words (code switching)"""
    soup = BeautifulSoup(line, features="html.parser")
    # Find all foreign tags which indicates English words
    foreign_tags = soup.find_all("foreign", lang="English")
    cs = 0
    sentence = []
    if foreign_tags:
        cs = 1
        # Create a list of labels for each English and Spanish word
        lids = []
        # Iterate over each token
        tag_tok = [
            clean(tag.get_text(), rm_punc, lc).strip().split() for tag in foreign_tags
        ]
        tag_tok = [
            item
            for sublist in tag_tok
            for item in sublist
            if item not in ["()", "(())"]
        ]
        for token in clean(soup.text, rm_punc, lc).split():
            # If the token is an English word, it will be surrounded by a foreign tag
            # We search for it in the list of English words we found earlier

            if token in tag_tok:
                lid = "en"
            else:
                lid = "es"
            if len(lids) == 0:
                lids.append(lid)
            if lid != lids[-1]:
                lids.append(lid)
                sentence.append("|")
            sentence.append(token)
        sentence = " ".join(sentence)
    else:
        lids = ["es"]
        sentence = soup.get_text()
    # Remove the foreign tags
    # sentence = soup.get_text()
    return sentence, lids, cs


# def floor_(number, decimals=8):
#     factor = 10 ** decimals
#     return math.floor(number * factor) / factor


def fix_small_errors(line) -> str:
    """The data has some small errors to fix
    https://github.com/apple/ml-code-switched-speech-translation/blob/main/fisher/extract_cs_words_from_raw_data.py
    """
    if 'lang+"English"' in line:
        line = line.replace('lang+"English"', 'lang="English"')
    if 'lan="English"' in line:
        line = line.replace('lan="English"', 'lang="English"')
    if " /foreign>" in line:
        line = line.replace(" /foreign>", "</foreign>")
    if '<foreign lang="English"> meeting <foreign lang="English">' in line:
        line = line.replace(
            '<foreign lang="English"> meeting <foreign lang="English">',
            '<foreign lang="English"> meeting </foreign>',
        )

    return line


def clean(text, rm_punc=False, lc=False):
    text = extra_cleaning(text)
    text = rm_noisy_punc(text)
    if rm_punc:
        text = remove_punc(text)
    if lc:
        text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()

    return text


def extra_cleaning(text):
    text = text.replace("\r", "")  # remove carriage returns

    # remove extra spaces around parentheses
    text = re.sub(r"\(", " (", text)
    text = re.sub(r"\)", ") ", text)
    text = re.sub(r" +", " ", text)  # remove multiple spaces
    text = re.sub(r"\) ([\.\!\:\?\;\,])", r")\1", text)
    text = re.sub(r"\( ", "(", text)
    text = re.sub(r" \)", ")", text)
    text = re.sub(r"(\d) %", r"\1%", text)
    text = re.sub(r" :", ":", text)
    text = re.sub(r" ;", ";", text)

    # normalize unicode punctuation
    text = re.sub(r"\`", "'", text)
    text = re.sub(r"\'\'", ' "', text)

    text = text.replace("„", '"')
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    text = text.replace("–", "-")
    text = text.replace("—", " - ")
    text = re.sub(r" +", " ", text)
    text = text.replace("´", "'")
    text = re.sub(r"([a-z])‘([a-z])", r"\1'\2", text, flags=re.I)
    text = re.sub(r"([a-z])’([a-z])", r"\1'\2", text, flags=re.I)
    text = text.replace("‘", "'")
    text = text.replace("‚", "'")
    text = text.replace("’", '"')
    text = text.replace("''", '"')
    text = text.replace("´´", '"')
    text = text.replace("…", "...")

    # handle pseudo-spaces
    text = re.sub(r" %", "%", text)
    text = re.sub(r"nº ", "nº ", text)
    text = re.sub(r" :", ":", text)
    text = re.sub(r" ºC", " ºC", text)
    text = re.sub(r" cm", " cm", text)
    text = re.sub(r" \?", "?", text)
    text = re.sub(r" \!", "!", text)
    text = re.sub(r" ;", ";", text)
    text = re.sub(r", ", ", ", text)
    text = re.sub(r" +", " ", text)

    # Spanish "quotation", followed by comma, style
    text = re.sub(r",\"", '",', text)
    text = re.sub(
        r"(\.+)\"(\s*[^<])", r"\"\1\2", text
    )  # don't fix period at end of sentence

    # Digit grouping
    text = re.sub(r"(\d) (\d)", r"\1,\2", text)

    return text


def rm_noisy_punc(line):

    # line = re.sub(r'\([^)]+\)', ' ', line)
    # Remove brackets and keep their contents
    line = re.sub(r"\(+|\)+|\(|\)", "", line)

    # Normalize punctuation
    line = line.replace("_", " ")
    line = line.replace("`", "'")  # for English
    line = line.replace("´", "'")  # for English
    line = line.replace("\¨", "'")  # I¨m -> I'm etc.

    # Remove noisy parts
    line = line.replace("noise", "")
    line = line.replace("laughter", "")
    line = line.replace("background noise", "")
    line = line.replace("background speech", "")

    # Specific replacements for different datasets
    line = line.replace("i/he", "i")

    # Remove noisy punctuation
    for char in "-()<>[]{}\\/;~=*":
        line = line.replace(char, " ")

    # Remove noisy punctuation at the beginning of a line
    line = re.sub(r"^[,.!]+", "", line)

    # Remove consecutive whitespaces
    line = re.sub(r"\s+", " ", line)

    # Remove the first and last whitespaces
    line = line.strip()

    # Print the cleaned line
    return line


def remove_punc(text):
    """This function removes all English punctuations except the single quote (verbatim)."""

    english_punctuations = string.punctuation + "¿¡"
    # # Remove the single quote from the punctuations as it is verbatim
    english_punctuations = english_punctuations.replace("'", "")

    # Create a translation table that maps each punctuation to a space.
    translator = str.maketrans("", "", english_punctuations)

    # Translate the text using the translation table
    text = text.translate(translator)

    return text
