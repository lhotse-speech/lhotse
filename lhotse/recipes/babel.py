"""
BABEL is a collection of corpora created during the IARPA BABEL program:
https://www.iarpa.gov/index.php/research-programs/babel

It has about 25 languages with 40h - 160h of training recordings and ~20h
of development set recordings.
"""
import logging
import re
from collections import defaultdict
from typing import Optional

from cytoolz import sliding_window

from lhotse import Recording, RecordingSet, SupervisionSegment, SupervisionSet, validate_recordings_and_supervisions
from lhotse.utils import Pathlike

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

OOV_PATTERN = re.compile(r'(\(\(\)\)|<foreign>|<prompt>|<overlap>|<hes>)')
SPK_NOISE_PATTERN = re.compile(r'<(limspack|lipsmack|breath|cough)>')
NOISE_PATTERN = re.compile(r'<(click|ring|dtmf|int|sta)>')
SIL_PATTERN = re.compile(r'<no-speech>')
REMOVE_PATTERN = re.compile(r'<(male-to-female|female-to-male)> ')


def prepare_single_babel_language(corpus_dir: Pathlike, output_dir: Optional[Pathlike] = None):
    manifests = defaultdict(dict)
    for split in ('dev', 'eval', 'training'):
        audio_dir = corpus_dir / f'conversational/{split}/audio'
        recordings = RecordingSet.from_recordings(Recording.from_sphere(p) for p in audio_dir.glob('*.sph'))
        if len(recordings) == 0:
            logging.warning(f"No SPHERE files found in {audio_dir}")
        manifests[split]['recordings'] = recordings

        supervisions = []
        text_dir = corpus_dir / f'conversational/{split}/transcription'
        for p in text_dir.glob('*'):
            # p.stem -> BABEL_BP_101_10033_20111024_205740_inLine
            # parts:
            #   0 -> BABEL
            #   1 -> BP
            #   2 -> <language-code> (101)
            #   3 -> <speaker-id> (10033)
            #   4 -> <date> (20111024)
            #   5 -> <hour> (205740)
            #   6 -> channel (inLine) ; inLine <=> A ; outLine <=> B ; "scripted" <=> A
            p0, p1, lang_code, speaker, date, hour, channel, *_ = p.stem.split('_')
            channel = {'inLine': 'A', 'outLine': 'B'}.get(channel, 'A')
            # Add a None at the end so that the last timestamp is only used as "next_timestamp"
            # and ends the iretation (otherwise we'd lose the last segment).
            lines = p.read_text().splitlines() + [None]
            for (timestamp, text), (next_timestamp, _) in sliding_window(2, zip(lines[::2], lines[1::2])):
                start = float(timestamp[1:-1])
                end = float(next_timestamp[1:-1])
                supervisions.append(
                    SupervisionSegment(
                        id=f'{lang_code}_{speaker}_{channel}_{date}_{hour}_{int(100 * start):06}',
                        recording_id=p.stem,
                        start=start,
                        duration=round(end - start, ndigits=8),
                        channel=0,
                        text=normalize_text(text),
                        language=BABELCODE2LANG[lang_code],
                        speaker=speaker,
                    )
                )
        if len(supervisions) == 0:
            logging.warning(f"No supervisions found in {text_dir}")
        manifests[split]['supervisions'] = SupervisionSet.from_segments(supervisions)

        validate_recordings_and_supervisions(
            manifests[split]['recordings'],
            manifests[split]['superevisions']
        )

        if output_dir is not None:
            language = BABELCODE2LANG[lang_code]
            if split == 'training':
                split = 'train'
            manifests[split]['recordings'].to_json(f'recordings_{language}_{split}.json')
            manifests[split]['supervisions'].to_json(f'supervisions_{language}_{split}.json')

    return manifests


def normalize_text(text: str) -> str:
    text = OOV_PATTERN.sub('<unk>', text)
    text = SPK_NOISE_PATTERN.sub('<v-noise>', text)
    text = NOISE_PATTERN.sub('<noise>', text)
    text = SIL_PATTERN.sub('<silence>', text)
    text = REMOVE_PATTERN.sub('', text)
    return text
