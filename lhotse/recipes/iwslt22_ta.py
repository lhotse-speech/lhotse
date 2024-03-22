# Copyright    2023  Johns Hopkins        (authors: Amir Hussein, Matthew Wiesner)

"""
The IWSLT Tunisian dataset is a 3-way parallel dataset consisting of approximately 160 hours
and 200,000 lines of aligned audio, Tunisian transcripts, and English translations. This dataset
comprises conversational telephone speech recorded at a sampling rate of 8kHz. The train, dev,
and test1 splits of the iwslt2022 shared task correspond to catalog number LDC2022E01. Please
note that access to this data requires an LDC subscription from your institution.To obtain this
dataset, you should download the predefined splits by running the following command:
git clone https://github.com/kevinduh/iwslt22-dialect.git. For more detailed information about
the shared task, please refer to the task paper available at this link:
https://aclanthology.org/2022.iwslt-1.10/.
"""

import logging
import re
import string
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import tqdm

from lhotse import (
    AudioSource,
    Recording,
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
    validate_recordings_and_supervisions,
)
from lhotse.qa import fix_manifests
from lhotse.utils import Pathlike

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


def download_iwslt22_ta(
    target_dir: Pathlike = ".",
) -> None:
    """
    Download and untar the dataset.

    NOTE: This function just returns with a message since IWSLT22 Tunisian-English is not available
    for direct download.

    :param target_dir: Pathlike, the path of the dir to storage the dataset.
    """
    logging.info(
        """
        To obtain this data your institution needs to have an LDC subscription.
        You also should download the pre-defined splits with
        git clone https://github.com/kevinduh/iwslt22-dialect.git
    """
    )


def prepare_iwslt22_ta(
    corpus_dir: Pathlike,
    splits: Pathlike,
    output_dir: Optional[Pathlike] = None,
    normalize_text: bool = False,
    langs: Optional[List[str]] = ["ta", "eng"],
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Prepares manifests for the train dev and test1 splits.

    :param corpus_dir: Path to ``LDC2022E01`` the path of the data dir.
    :param splits: Path to splits from https://github.com/kevinduh/iwslt22-dialect
    :param normalize_text: Bool, if True, Arabic text normalization is performed
        from https://aclanthology.org/2022.iwslt-1.29.pdf.
    :param output_dir: Directory where the manifests should be written. Can be omitted
        to avoid writing.
    :param langs: str, list of language abbreviations for source and target languages.
    :param num_jobs: int, the number of jobs to use for parallel processing.
    :return: A dict with manifests. The keys are: ``{'recordings', 'supervisions'}``.

    """
    import soundfile as sf

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
    futures = []
    with ThreadPoolExecutor(num_jobs) as ex:
        for p in text_dir.glob("*.tsv"):
            if not p.stem.startswith("._"):
                translations_path = (
                    p.parent.parent.parent / "translations" / "ta" / p.name
                )
                translations_path = translations_path.with_name(
                    p.stem.split(".")[0] + ".eng" + p.suffix
                )
                if translations_path.exists():
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
                            translations_path,
                            normalize_text,
                            exclude,
                            langs,
                        )
                    )
                else:
                    logging.warning(
                        f"{translations_path.stem} does not exist, please make sure number of translations = transcriptions"
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
        recordings, supervisions = fix_manifests(recordings, supervisions)
        validate_recordings_and_supervisions(recordings, supervisions)
        for split in ("train", "dev", "test1"):
            sups_ = supervisions.filter(lambda s: s.recording_id in split_files[split])
            recs_ = recordings.filter(lambda r: r.id in split_files[split])
            manifests[split] = {"recordings": recs_, "supervisions": sups_}

            if output_dir is not None:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                sups_.to_file(output_dir / f"iwslt22-ta_recordings_{split}.jsonl.gz")
                recs_.to_file(output_dir / f"iwslt22-ta_supervisions_{split}.jsonl.gz")

    return manifests


def _filename_to_supervisions(
    p: Path,
    translations_path: Path,
    normalize_text: bool,
    exclude: list,
    langs: list,
):
    supervisions = []
    date, time, someid, channel = p.with_suffix("").stem.split("_")
    date_tgt, time_tgt, someid_tgt, channel_tgt = p.with_suffix("").stem.split("_")
    translations = translations_path.read_text().splitlines()
    transcripts = p.read_text().splitlines()

    sorted_translations = sorted(translations, key=lambda line: line.split("\t")[0])
    sorted_transcripts = sorted(transcripts, key=lambda line: line.split("\t")[0])

    for src, tgt in zip(sorted_transcripts, sorted_translations):

        start, end, sid, text = src.rstrip().split("\t")
        _, _, _, text_tgt = tgt.rstrip().split("\t")
        start = float(start)
        end = float(end)

        # Following the IWSLT provided text normalization
        text = normalize_text(text, "transcript")
        text_tgt = normalize_text(text_tgt, "translation")

        utt_id = f"{date}_{time}_{someid}_{channel}_{int(100*start):06}"
        utt_id_tgt = (
            f"{date_tgt}_{time_tgt}_{someid_tgt}_{channel_tgt}_{int(100*start):06}"
        )

        assert (
            utt_id == utt_id_tgt
        ), f"The loaded source and target files are not sorted properly: {utt_id} {utt_id_tgt}"

        if normalize_text:
            # Aggressive Tunisian text normalization from https://aclanthology.org/2022.iwslt-1.29.pdf
            text = text_cleaning(text)
            if text.strip() == "":
                logging.warning(
                    f"Skipping {p.stem} {start} {end} {text} with empty cleaned transcript ..."
                )
                continue

        if utt_id in exclude:
            continue
        supervisions.append(
            SupervisionSegment(
                id=f"{sid}_{langs[0]}_{langs[1]}_{utt_id}",
                recording_id=p.with_suffix("").stem,
                start=start,
                duration=round(end - start, ndigits=8),
                channel=0,
                text=text,
                language=langs[0],
                speaker=sid,
                custom={"translated_text": {langs[1]: text_tgt}},
            )
        )
    return supervisions


def normalize_text(utterance, language):
    arabic_filter = re.compile(r"[OUM]+/*|\u061F|\?|\!|\.")
    english_filter = re.compile(r"\(|\)|\#|\+|\=|\?|\!|\;|\.|\,|\"|\:")
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


def text_cleaning(text):
    text = remove_punctuations(text)
    text = east_to_west_num(text)
    text = remove_diacritics(text)
    text = remove_extra_space(text)
    text = normalize_arabic(text)
    text = normalize_text_(text)
    return text
