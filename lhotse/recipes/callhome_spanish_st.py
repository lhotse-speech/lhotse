"""
About the CALLHOME Spanish corpus

    This is conversational telephone speech collected as 2-channel μ-law, 8kHz-sampled data. 
    The catalog number LDC96S35 for audio corpus and LDC96T17 for transcripts.

    This data is not available for free - your institution needs to have an LDC subscription.
    You should also download and prepare the pre-defined splits with:
        git clone https://github.com/joshua-decoder/fisher-callhome-corpus.git
        cd fisher-callhome-corpus
        make
        cd ../
"""

from decimal import Decimal
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
    split_map: Dict[Tuple[str, str], Tuple[str, str, List[int]]],
    splits_dir: Pathlike,
    trans_path: Pathlike,
    split: str,
    remove_punc: bool,
    lowercase: bool,
) -> List[SupervisionSegment]:

    splits_path = splits_dir / "mapping" / split
    trans_split = split.split("_")[1]
    trans_path = trans_path / "transcrp" / trans_split

    # load info: duration, translation, channel
    txt_files = {}
    idx = 0
    mapping_file = open(splits_path, "r").readlines()
    segments = []
    with tqdm(total=len(mapping_file), desc=f"Create {split} supervisions") as pbar:
        for l in mapping_file:
            file_name, _ = l.strip().split()
            if file_name not in txt_files:
                txt_files[file_name] = [
                    [word.replace(":", "") for word in line.strip().split()]
                    for line in open(
                        trans_path / f"{file_name}.txt", "r", encoding="ISO-8859-1"
                    ).readlines()
                ]

            file = txt_files[file_name]
            text_es, text_en, line_num = split_map[(file_name, idx)]

            text_es = clean(text_es, remove_punc, lowercase)
            text_en = clean(text_en, remove_punc, lowercase)
            s_time = float(file[line_num[0] - 1][0])
            e_time = float(file[line_num[-1] - 1][1])
            spk = file[line_num[0] - 1][2]
            if "A" in spk:
                spk = "A"
            elif "B" in spk:
                spk = "B"
            segments.append(
                SupervisionSegment(
                    id=file_name + "-" + str(idx).zfill(len(str(len(split_map)))),
                    recording_id=file_name,
                    start=s_time,
                    duration=float(Decimal(e_time) - Decimal(s_time)),
                    channel=ord(spk) - ord("A"),
                    text=text_es,
                    language=["es"],
                    speaker=f"{file_name}_{spk}",
                    custom={"translated_text": {"en": text_en}},
                )
            )
            idx += 1
            pbar.update()
    return segments


def prepare_callhome_spanish(
    audio_dir_path: Pathlike,
    transcript_dir_path: Pathlike,
    split_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    absolute_paths: bool = True,
    remove_punc: bool = True,
    lowercase: bool = True,
    num_jobs: int = 4,
) -> Dict[str, Union[RecordingSet, SupervisionSet]]:
    """
    Prepares manifests for Callhome Spanish
    :param audio_dir_path: Path to audio directory  (LDC96S35).
    :param transcript_dir_path: Path to transcript directory (LDC96T17).
    :param split_dir: Path to splits from https://github.com/joshua-decoder/fisher-callhome-corpus.git
    :param output_dir: Directory where the manifests should be written. Can be omitted to avoid writing.
    :param absolute_paths: Whether to return absolute or relative (to the corpus dir) paths for recordings.
    :param remove_punc: Remove punctuations from the text
    :param lowercase: Lower case the text
    :return: A dict with manifests. The keys are: ``{'recordings', 'supervisions'}``.
    """

    manifests = {}
    split_map = load_splits(split_dir)
    audio_dir_path, transcript_dir_path = Path(audio_dir_path), Path(
        transcript_dir_path
    )

    audio_paths = check_and_rglob(audio_dir_path, "*.sph")

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
        supervisions = create_supervision(
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
                output_dir / f"callhome-spanish_recordings_{split_set}.jsonl.gz"
            )
            supervisions.to_file(
                output_dir / f"callhome-spanish_supervisions_{split_set}.jsonl.gz"
            )
        manifests[split] = {"recordings": recordings_tmp, "supervisions": supervisions}
    return manifests


def load_splits(splits_dir):
    splits = {}
    for split in ("callhome_devtest", "callhome_evltest", "callhome_train"):
        mapping = {}
        splits_path = splits_dir / "mapping" / split

        # load transcriptions (text_es) and tranlations (text_en)
        transcription_file = splits_dir / "corpus" / "ldc" / f"{split}.es"

        translation_file = splits_dir / "corpus" / "ldc" / f"{split}.en"
        transcriptions = [line.strip() for line in open(transcription_file).readlines()]
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
