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
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Union
from pathlib import Path

from cytoolz import sliding_window

import tarfile
import shutil
from lhotse import Recording, RecordingSet, SupervisionSegment, SupervisionSet, validate_recordings_and_supervisions
from lhotse.qa import remove_missing_recordings_and_supervisions, trim_supervisions_to_recordings
from lhotse.utils import Pathlike, urlretrieve_progress
import regex as re2
from functools import partial
import unicodedata


# Keep Markings such as vowel signs, all letters, and decimal numbers 
VALID_CATEGORIES = ('Mc', 'Mn', 'Ll', 'Lm', 'Lo', 'Lt', 'Lu', 'Nd', 'Zs')
noise_pattern = re2.compile(r'\([^)]*\)', re2.UNICODE)
apostrophe_pattern = re2.compile(r"(\w)'(\w)")
html_tags = re2.compile(r"(&[^ ;]*;)|(</?[iu]>)")
KEEP_LIST = [u'\u2019']


ASR = ('es', 'fr', 'pt', 'it', 'ru', 'el', 'ar', 'de',)


ISOCODE2LANG = {
    'fr': 'French',
    'es': 'Spanish',
    'pt': 'Portuguese',
    'it': 'Italian',
    'ru': 'Russian',
    'el': 'Greek',
    'ar': 'Arabic',
    'de': 'German',
}


def download_and_untar(
    target_dir: Pathlike = '.',
    force_download: Optional[bool] = False
    language: str = 'fr',
) -> None:
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    tar_path = target_dir / f'{language}-{language}.tgz'
    urlretreive_progress(
        f'http://www.openslr.org/resources/100/mtedx_{language}-{language}.tgz',
        filename=tar_path,
        desc=f'Downloading MTEDx {language}',
    )
    corpus_dir = target_dir / f'{language}-{language}.tgz'
    completed_detector = corpus_dir / '.completed'
    if not completed_detector.is_file():
        shutil.rmtree(corpus_dir, ignore_errors=True)
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=target_dir)
            completed_detector.touch()


def prepare_single_mtedx_language(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Prepares manifests using a single MTEDx language.

    This function works as follows:

        - First it looks for the audio directory in the data/wav where the .flac
            files are stored.
        - Then, it looks for the vtt directory in data/{train,dev,test}/vtt
            which contains the segmentation and transcripts for the audio.

    :param corpus_dir: Path to the root of the MTEDx download
    :param output_dir: Path where the manifests are stored as .json files
    :return:
    """
    if isinstance(corpus_dir, str):
        corpus_dir = Path(corpus_dir)
    manifests = defaultdict(dict)
    language = ISOCODE2LANG[corpus_dir.stem.split('-')[0]]
    filterfun = partial(_filter_word)

    for split in ('train', 'valid', 'test'):
        audio_dir = corpus_dir / f'data/{split}/wav'
        recordings = RecordingSet.from_recordings(Recording.from_file(p) for p in audio_dir.glob('*.flac'))
        if len(recordings) == 0:
            logging.warning(f'No .flac files found in {audio_dir}')
        
        supervisions = []
        text_dir = corpus_dir / f'data/{split}/vtt'
        for p in text_dir.glob('*'):
            lines = p.read_text()            
            recoid = p.stem.split('.')[0]
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
                    line_ = ' '.join(line_list)
                if re.match(r"^\w+ *(<[^>]*> *)+$", line_, re.UNICODE):
                    line_new = line_.strip()
                elif "<" in line_ or ">" in line_:
                    continue;
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
    
        if len(supervisions) == 0:
            logging.warning(f'No supervisions found in {text_dir}')
        supervisions = SupervisionSet.from_segments(supervisions)
        
        recordings, supervisions = remove_missing_recordings_and_supervisions(recordings, supervisions)
        supervisions = trim_supervisions_to_recordings(recordings, supervisions)
        validate_recordings_and_supervisions(recordings, supervisions)

        manifests[split] = {
            'recordings': recordings,
            'supervisions': supervisions,
        }

        if output_dir is not None:
            if isinstance(output_dir, str):
                output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            save_split = 'dev' if split == 'valid' else split
            recordings.to_file(output_dir / f'recordings_{language}_{split}.json')
            supervisions.to_file(output_dir / f'supervisions_{language}_{split}.json')

    return dict(manifests)
    

def _format_uttid(recoid, start):
    # Since each recording is a talk, normally by 1 speaker, we use the
    # recoid as the spkid as well. 
    start = '{0:08d}'.format(int(float(start)*100))
    return '_'.join([recoid, start])
 

def _filter_word(s):
    for c in s:
        if unicodedata.category(c) not in VALID_CATEGORIES and c not in KEEP_LIST:
            return False
    return True


def _filter(s):
    return unicodedata.category(s) in VALID_CATEGORIES or s in KEEP_LIST 


def _time2sec(time):
    hr, mn, sec = time.split(':')
    return int(hr) * 3600.0 + int(mn) * 60.0 + float(sec)


def _parse_time_segment(l):
    start, end = l.split(' --> ')
    start = _time2sec(start)    
    end = _time2sec(end)
    return start, end


def _normalize_space(c):
    if unicodedata.category(c) == 'Zs':
        return " "
    else:
        return c


def _parse_vtt(lines, noise):
    blocks = lines.split('\n\n') 
    for i, b in enumerate(blocks, -1):
        if i > 0 and b.strip() != "":
            b_lines = b.split('\n')
            start, end = _parse_time_segment(b_lines[0])
            line = ' '.join(b_lines[1:])
            line_new = line
            if line.strip('- ') != '':
                line_parts = noise_pattern.sub(noise, line_new)
                line_parts = apostrophe_pattern.sub(r"\1\u2019\2", line_parts)
                line_parts = html_tags.sub('', line_parts)
                line_parts_new = []
                for lp in line_parts.split(noise):
                    line_parts_new.append(
                        ''.join(
                            [i for i in filter(_filter, lp.strip().replace('-', ' '))] 
                        )
                    )
                joiner = ' ' + noise + ' '
                line_new = joiner.join(line_parts_new)
                line_new = re2.sub(r"\p{Zs}", lambda m: _normalize_space(m.group(0)), line_new)
                line_new = re2.sub(r' +', ' ', line_new).strip().lower()
            yield start, end, line_new

