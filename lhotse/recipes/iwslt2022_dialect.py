"""
LDC2022E01 (Tunisian Arabic Dialect) is a speech translation corpus
LDC2022E02 (Tunisian Arabic Dialect Eval Set)
"""
import logging
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Union
from pathlib import Path
from concurrent.futures.thread import ThreadPoolExecutor
import tqdm
import soundfile as sf
import string

try:
    import pyarabic.number as number
    from pyarabic import araby
except ImportError:
    import pip

    pip._internal.main(["install", "pyarabic"])
    import pyarabic.number as number
    from pyarabic import araby

from cytoolz import sliding_window

from lhotse import (
    Recording,
    AudioSource,
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
    validate_recordings_and_supervisions,
)
from lhotse.qa import (
    remove_missing_recordings_and_supervisions,
    trim_supervisions_to_recordings,
)
from lhotse.utils import Pathlike
from lhotse.manipulation import combine


# English annotation rules:
# -------------------------------
#     (()) - Uncertain word or words
#      %pw - Partial word
#       #  - Foreign word, either followed by translation or (()) if cannot translate
#       +  - Mis-pronounced word (carried over from mispronunciation marked in transcript)
#      uh, um, eh or ah - Filled pauses
#       =  - Typographical error from transcript


# Arabic annotation rules:
# -------------------------------
# O/ - foreign
# U/ - uncertain
# M/ - MSA
# UM/ - uncertain + MSA
# UO/ - uncertain + foreign


arabic_filter = re.compile(r"[OUM]+/*|\u061F|\?|\!|\.")
english_filter = re.compile(r"\(|\)|\#|\+|\=|\?|\!|\;|\.|\,|\"|\:")


def prepare_iwslt2022_dialect_eval(
    corpus_dir: Pathlike,
    segments: Pathlike,
    output_dir: Optional[Pathlike] = None,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Prepares manifests for the eval splits.
    """
    manifests = {}
    corpus_dir = Path(corpus_dir)
    audio_dir = corpus_dir / "data" / "audio"
    recordings = {}
    supervisions = []
    for p in audio_dir.glob("*.sph"):
        audio_sf = sf.SoundFile(str(p))
        filename = p.stem
        if filename not in recordings:
            recordings[filename] = Recording(
                id=filename,
                sources=[
                    AudioSource(
                        type="file",
                        channels=[0],
                        source=str(p),
                    ),
                ],
                sampling_rate=audio_sf.samplerate,
                num_samples=audio_sf.frames,
                duration=audio_sf.frames / audio_sf.samplerate,
            )

    with open(segments) as f:
        for l in f:
            uttid, recoid, start, end = l.strip().split()
            start, end = float(start), float(end)
            sid = uttid.split("_")[2]
            supervisions.append(
                SupervisionSegment(
                    id=f"{uttid}",
                    recording_id=recoid,
                    start=start,
                    duration=round(end - start, ndigits=8),
                    channel=0,
                    text="",
                    language="transcript",
                    speaker=sid,
                )
            )
    supervisions = SupervisionSet.from_segments(supervisions)
    recordings = RecordingSet.from_recordings(recordings.values())
    recordings, supervisions = remove_missing_recordings_and_supervisions(
        recordings,
        supervisions,
    )
    supervisions = trim_supervisions_to_recordings(recordings, supervisions)
    validate_recordings_and_supervisions(recordings, supervisions)
    manifests["eval"] = {"recordings": recordings, "supervisions": supervisions}

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        recordings.to_file(output_dir / f"recordings_eval.json")
        supervisions.to_file(output_dir / f"supervisions_eval.json")

    return manifests


def prepare_iwslt2022_dialect(
    corpus_dir: Pathlike,
    splits: Pathlike,
    output_dir: Optional[Pathlike] = None,
    cleaned: bool = False,
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Prepares manifests for the train dev and test1 splits.
    """
    manifests = {}
    split_files = load_splits(Path(splits))
    corpus_dir = Path(corpus_dir)
    audio_dir = corpus_dir / "data/audio/ta"
    exclude = []
    with open(str(Path(splits) / "exclude-utterance.txt")) as f:
        for l in f:
            excludeid, start, end = l.strip().split()
            start = float(start)
            exclude.append(f"{excludeid}_{int(100*start):06}")

    recordings = {}
    supervisions = []
    text_dir = corpus_dir / "data/transcripts/ta"
    translation_dir = corpus_dir / "data/translations/ta"
    futures = []
    with ThreadPoolExecutor(num_jobs) as ex:
        for dir_type, lbl in zip(
            (text_dir, translation_dir), ("transcript", "translation")
        ):
            for p in dir_type.glob("*.tsv"):
                if not p.stem.startswith("._"):
                    filename = p.with_suffix("").stem
                    audio_sf = sf.SoundFile(str(audio_dir / f"{filename}.sph"))
                    if filename not in recordings:
                        recordings[filename] = Recording(
                            id=filename,
                            sources=[
                                AudioSource(
                                    type="file",
                                    channels=[0],
                                    source=str(audio_dir / f"{filename}.sph"),
                                ),
                            ],
                            sampling_rate=audio_sf.samplerate,
                            num_samples=audio_sf.frames,
                            duration=audio_sf.frames / audio_sf.samplerate,
                        )
                    futures.append(
                        ex.submit(
                            _filename_to_supervisions,
                            p,
                            cleaned,
                            split_files["train"],
                            exclude,
                            lbl,
                        )
                    )

        for future in tqdm.tqdm(futures, desc="Processing text", leave=False):
            result = future.result()
            if result is None:
                continue
            for sup in result:
                supervisions.append(sup)

        supervisions = deduplicate_supervisions(supervisions)
        supervisions = SupervisionSet.from_segments(supervisions)
        recordings = RecordingSet.from_recordings(recordings.values())
        recordings, supervisions = remove_missing_recordings_and_supervisions(
            recordings,
            supervisions,
        )
        supervisions = trim_supervisions_to_recordings(recordings, supervisions)
        validate_recordings_and_supervisions(recordings, supervisions)
        for ds in ("train", "dev", "test1"):
            sups_ds = supervisions.filter(lambda s: s.recording_id in split_files[ds])
            recos_ds = recordings.filter(lambda r: r.id in split_files[ds])
            manifests[ds] = {"recordings": recos_ds, "supervisions": sups_ds}

            if output_dir is not None:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                recos_ds.to_file(output_dir / f"recordings_{ds}.json")
                sups_ds.to_file(output_dir / f"supervisions_{ds}.json")

    return manifests


def _filename_to_supervisions(
    p: Path,
    cleaned: bool,
    train: list,
    exclude: list,
    label: str,
):
    supervisions = []
    date, time, someid, channel = p.with_suffix("").stem.split("_")
    for l in p.read_text().splitlines():
        start, end, sid, text = l.rstrip().split("\t")
        start = float(start)
        end = float(end)
        text = normalize_text(text, label)
        if cleaned and label == "transcript":
            text = data_cleaning(text)
            if text.strip() == "" and p.stem in train:
                logging.warning(
                    f"Skipping {p.stem} {start} {end} {text} with empty cleaned transcript ..."
                )
                continue
            elif text.strip() == "":
                text = "<noise>"

        utt_id = f"{date}_{time}_{someid}_{channel}_{int(100*start):06}"
        if utt_id in exclude:
            continue
        lang = "tun" if label == "transcript" else "eng"
        supervisions.append(
            SupervisionSegment(
                id=f"{sid}_{lang}_{utt_id}",
                recording_id=p.with_suffix("").stem,
                start=start,
                duration=round(end - start, ndigits=8),
                channel=0,
                text=text,
                language=label,
                speaker=sid,
            )
        )
    return supervisions


def normalize_text(utterance, language):
    if language == "transcript":
        return re.subn(arabic_filter, "", utterance)[0]
    elif language == "translation":
        return re.subn(english_filter, "", utterance)[0].lower()
    else:
        raise ValueError(f"Text normalization for {language} is not supported")


def load_splits(path):
    splits = {}
    for split in ("train", "dev", "test1"):
        file_list = []
        split_scp = path / f"{split}.file_id.txt"
        with open(str(split_scp)) as f:
            for l in f:
                file_list.append(l.strip())
        splits[split] = file_list
    return splits


def deduplicate_supervisions(
    supervisions: Iterable[SupervisionSegment],
) -> List[SupervisionSegment]:
    from cytoolz import groupby

    duplicates = groupby((lambda s: s.id), sorted(supervisions, key=lambda s: s.id))
    filtered = []
    for k, v in duplicates.items():
        if len(v) > 1:
            logging.warning(
                f"Found {len(v)} supervisions with conflicting IDs ({v[0].id}) "
                f"- keeping only the first one."
            )
        filtered.append(v[0])
    return filtered


_unicode = "\u0622\u0624\u0626\u0628\u062a\u062c\u06af\u062e\u0630\u0632\u0634\u0636\u0638\u063a\u0640\u0642\u0644\u0646\u0648\u064a\u064c\u064e\u0650\u0652\u0670\u067e\u0686\u0621\u0623\u0625\u06a4\u0627\u0629\u062b\u062d\u062f\u0631\u0633\u0635\u0637\u0639\u0641\u0643\u0645\u0647\u0649\u064b\u064d\u064f\u0651\u0671"
_buckwalter = "|&}btjGx*z$DZg_qlnwyNaio`PJ'><VApvHdrsSTEfkmhYFKu~{"
_backwardMap = {ord(b): a for a, b in zip(_buckwalter, _unicode)}


def from_buckwalter(s):
    return s.translate(_backwardMap)


def read_tsv(f):
    text_data = list()
    for line in f:
        if not line.strip():
            continue
        text_data.append(line.strip().split("\t"))
    return text_data


_preNormalize = " \u0629\u0649\u0623\u0625\u0622"
_postNormalize = " \u0647\u064a\u0627\u0627\u0627"
_toNormalize = {ord(b): a for a, b in zip(_postNormalize, _preNormalize)}


def normalize_text_(s):
    return s.translate(_toNormalize)


def normalize_arabic(text):
    text = re.sub("[إأٱآا]", "ا", text)
    text = re.sub(r"(أ){2,}", "ا", text)
    text = re.sub(r"(ا){2,}", "ا", text)
    text = re.sub(r"(آ){2,}", "ا", text)
    text = re.sub(r"(ص){2,}", "ص", text)
    text = re.sub(r"(و){2,}", "و", text)
    return text


def remove_english_characters(text):
    return re.sub(r"[^\u0600-\u06FF\s]+", "", text)


def remove_diacritics(text):
    # https://unicode-table.com/en/blocks/arabic/
    return re.sub(r"[\u064B-\u0652\u06D4\u0670\u0674\u06D5-\u06ED]+", "", text)


def remove_punctuations(text):
    """This function  removes all punctuations except the verbatim"""

    arabic_punctuations = """`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ"""
    english_punctuations = string.punctuation
    all_punctuations = set(
        arabic_punctuations + english_punctuations
    )  # remove all non verbatim punctuations

    for p in all_punctuations:
        if p in text:
            text = text.replace(p, " ")
    return text


def remove_extra_space(text):
    text = re.sub("\s+", " ", text)
    text = re.sub("\s+\.\s+", ".", text)
    return text


def remove_dot(text):
    words = text.split()
    res = []
    for word in words:
        word = re.sub("\.$", "", word)
        if word.replace(
            ".", ""
        ).isnumeric():  # remove the dot if it is not part of a number
            res.append(word)

        else:
            res.append(word)

    return " ".join(res)


def east_to_west_num(text):
    eastern_to_western = {
        "٠": "0",
        "١": "1",
        "٢": "2",
        "٣": "3",
        "٤": "4",
        "٥": "5",
        "٦": "6",
        "٧": "7",
        "٨": "8",
        "٩": "9",
        "٪": "%",
        "_": " ",
        "ڤ": "ف",
        "|": " ",
    }
    trans_string = str.maketrans(eastern_to_western)
    return text.translate(trans_string)


def digit2num(text, dig2num=False):

    """This function is used to clean numbers"""

    # search for numbers with spaces
    # 100 . 000 => 100.000

    res = re.search("[0-9]+\s\.\s[0-9]+", text)
    if res != None:
        t = re.sub(r"\s", "", res[0])
        text = re.sub(res[0], t, text)

    # seperate numbers glued with words
    # 3أشهر => 3 أشهر
    # من10الى15 => من 10 الى 15
    res = re.findall(r"[^\u0600-\u06FF\a-z]+", text)  # search for digits
    for match in res:
        if match not in {".", " "}:
            text = re.sub(match, " " + match + " ", text)
            text = re.sub("\s+", " ", text)

    # transliterate numbers to digits
    # 13 =>  ثلاثة عشر

    if dig2num == True:
        words = araby.tokenize(text)
        for i in range(len(words)):
            digit = re.sub(r"[\u0600-\u06FF]+", "", words[i])
            if digit.isnumeric():
                sub_word = re.sub(r"[^\u0600-\u06FF]+", "", words[i])
                if number.number2text(digit) != "صفر":
                    words[i] = sub_word + number.number2text(digit)
            else:
                pass

        return " ".join(words)
    else:
        return text


def seperate_english_characters(text):
    """
    This function separates the glued English and Arabic words
    """
    text = text.lower()
    res = re.findall(r"[a-z]+", text)  # search for english words
    for match in res:
        if match not in {".", " "}:
            text = re.sub(match, " " + match + " ", text)
            text = re.sub("\s+", " ", text)
    return text


def data_cleaning(text):
    text = remove_punctuations(text)
    text = east_to_west_num(text)
    text = seperate_english_characters(text)
    text = remove_diacritics(text)
    text = remove_extra_space(text)
    text = normalize_arabic(text)
    text = normalize_text_(text)
    text = digit2num(text, False)
    return text
