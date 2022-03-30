"""
BABEL is a collection of corpora created during the IARPA BABEL program:
https://www.iarpa.gov/index.php/research-programs/babel

It has about 25 languages with 40h - 160h of training recordings and ~20h
of development set recordings.
"""
import logging
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Union
from pathlib import Path
import tqdm

from cytoolz import sliding_window

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
from lhotse.utils import Pathlike
from lhotse.manipulation import combine

BABELCODE2LANG = {
    "101": "Cantonese",
    "102": "Assamese",
    "103": "Bengali",
    "104": "Pashto",
    "105": "Turkish",
    "106": "Tagalog",
    "107": "Vietnamese",
    "201": "Haitian",
    "202": "Swahili",
    "203": "Lao",
    "204": "Tamil",
    "205": "Kurmanji",
    "206": "Zulu",
    "207": "Tok-Pisin",
    "301": "Cebuano",
    "302": "Kazakh",
    "303": "Telugu",
    "304": "Lithuanian",
    "305": "Guarani",
    "306": "Igbo",
    "307": "Amharic",
    "401": "Mongolian",
    "402": "Javanese",
    "403": "Dholuo",
    "404": "Georgian",
}

OOV_PATTERN = re.compile(r"(\(\(\)\)|<foreign>|<prompt>|<overlap>|<hes>)")
SPK_NOISE_PATTERN = re.compile(r"<(limspack|lipsmack|breath|cough)>")
NOISE_PATTERN = re.compile(r"<(click|ring|dtmf|int|sta)>")
SIL_PATTERN = re.compile(r"<no-speech>")
REMOVE_PATTERN = re.compile(r"<(male-to-female|female-to-male)> ")


def prepare_single_babel_language(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    no_eval_ok: bool = False,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Prepares manifests using a single BABEL LDC package.

    This function works like the following:

        - first, it will scan `corpus_dir` for a directory named `conversational`;
            if there is more than once, it picks the first one (and emits a warning)
        - then, it will try to find `dev`, `eval`, and `training` splits inside
            (if any of them is not present, it will skip it with a warning)
        - finally, it scans the selected location for SPHERE audio files and transcripts.

    :param corpus_dir: Path to the root of the LDC package with a BABEL language.
    :param output_dir: Path where the manifests are stored.json
    :param no_eval_ok: When set to True, this function won't emit a warning
        that the eval set was not found.
    :return:
    """
    manifests = defaultdict(dict)

    # Auto-detect the location of the "conversational" directory
    orig_corpus_dir = corpus_dir
    corpus_dir = Path(corpus_dir)
    corpus_dir = [d for d in corpus_dir.rglob("conversational") if d.is_dir()]
    if not corpus_dir:
        raise ValueError(
            f"Could not find 'conversational' directory anywhere inside '{orig_corpus_dir}' "
            f"- please check your path."
        )
    if len(corpus_dir) > 1:
        # People have very messy data distributions, the best we can do is warn them.
        logging.warning(
            f"It seems there are multiple 'conversational' directories in '{orig_corpus_dir}' - "
            f"we are selecting the first one only ({corpus_dir[0]}). Please ensure that you provided "
            f"the path to a single language's dir, and the root dir for all BABEL languages."
        )
    corpus_dir = corpus_dir[0].parent

    for split in ("dev", "eval", "training"):
        audio_dir = corpus_dir / f"conversational/{split}/audio"
        sph_recordings = RecordingSet.from_recordings(
            Recording.from_file(p) for p in audio_dir.glob("*.sph")
        )
        wav_recordings = RecordingSet.from_recordings(
            Recording.from_file(p) for p in audio_dir.glob("*.wav")
        )
        recordings = combine(sph_recordings, wav_recordings)
        if len(recordings) == 0:
            if split == "eval" and no_eval_ok:
                continue
            logging.warning(f"No SPHERE or WAV files found in {audio_dir}")

        supervisions = []
        text_dir = corpus_dir / f"conversational/{split}/transcription"
        for p in tqdm.tqdm(text_dir.glob("*")):
            # p.stem -> BABEL_BP_101_10033_20111024_205740_inLine
            # parts:
            #   0 -> BABEL
            #   1 -> BP
            #   2 -> <language-code> (101)
            #   3 -> <speaker-id> (10033)
            #   4 -> <date> (20111024)
            #   5 -> <hour> (205740)
            #   6 -> channel (inLine) ; inLine <=> A ; outLine <=> B ; "scripted" <=> A
            p0, p1, lang_code, speaker, date, hour, channel, *_ = p.stem.split("_")
            channel = {"inLine": "A", "outLine": "B"}.get(channel, "A")
            # Fix problematic segments that have two consecutive timestamp lines with no transcript in between
            lines = p.read_text().splitlines() + [""]
            lines = [
                prev_l
                for prev_l, l in sliding_window(2, lines)
                if not (prev_l.startswith("[") and l.startswith("["))
            ]
            # Add a None at the end so that the last timestamp is only used as "next_timestamp"
            # and ends the iretation (otherwise we'd lose the last segment).
            lines += [None]
            for (timestamp, text), (next_timestamp, _) in sliding_window(
                2, zip(lines[::2], lines[1::2])
            ):
                try:
                    start = float(timestamp[1:-1])
                    end = float(next_timestamp[1:-1])
                    # Create supervision
                    supervisions.append(
                        SupervisionSegment(
                            id=f"{lang_code}_{speaker}_{channel}_{date}_{hour}_{int(100 * start):06}",
                            recording_id=p.stem,
                            start=start,
                            duration=round(end - start, ndigits=8),
                            channel=0,
                            text=normalize_text(text),
                            language=BABELCODE2LANG[lang_code],
                            speaker=f"{lang_code}_{speaker}_{channel}",
                        )
                    )
                except Exception as e:
                    logging.warning(f"Error while parsing segment. Message: {str(e)}")
                    raise ValueError(
                        f"Too many errors while parsing segments (file: '{p}'). "
                        f"Please check your data or increase the threshold."
                    )
        supervisions = deduplicate_supervisions(supervisions)

        if len(supervisions) == 0:
            logging.warning(f"No supervisions found in {text_dir}")
        supervisions = SupervisionSet.from_segments(supervisions)

        # Fixing and validation of manifests
        if split == "eval" and len(supervisions) == 0:
            # We won't remove missing recordings for the "eval" split in cases where
            # the user does not have its corresponding transcripts (very likely).
            pass
        else:
            recordings, supervisions = remove_missing_recordings_and_supervisions(
                recordings, supervisions
            )
            supervisions = trim_supervisions_to_recordings(recordings, supervisions)
        validate_recordings_and_supervisions(recordings, supervisions)

        manifests[split] = {"recordings": recordings, "supervisions": supervisions}

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            language = BABELCODE2LANG[lang_code]
            save_split = "train" if split == "training" else split
            recordings.to_file(output_dir / f"recordings_{language}_{save_split}.json")
            supervisions.to_file(
                output_dir / f"supervisions_{language}_{save_split}.json"
            )

    return dict(manifests)


def normalize_text(text: str) -> str:
    text = OOV_PATTERN.sub("<unk>", text)
    text = SPK_NOISE_PATTERN.sub("<v-noise>", text)
    text = NOISE_PATTERN.sub("<noise>", text)
    text = SIL_PATTERN.sub("<silence>", text)
    text = REMOVE_PATTERN.sub("", text)
    return text


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
