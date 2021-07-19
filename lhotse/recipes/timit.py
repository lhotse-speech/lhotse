#!/usr/bin/env python3

# Copyright 2021 Xiaomi Corporation (Author: Mingshuang Luo)
# Apache 2.0

import logging
import zipfile
from collections import defaultdict
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional, Union

from tqdm import tqdm

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, urlretrieve_progress


def download_timit(
        target_dir: Pathlike = '.',
        force_download: bool = False,
        base_url: Optional[str] = 'https://data.deepai.org/timit.zip') -> None:
    """
    Download and unzip the dataset TIMIT.
    :param target_dir: Pathlike, the path of the dir to store the dataset.
    :param force_download: bool, if True, download the zips no matter if the zips exists.
    :param base_url: str, the URL of the TIMIT dataset to download.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_name = 'timit.zip'
    zip_path = target_dir / zip_name
    corpus_dir = zip_path.with_suffix('')
    completed_detector = corpus_dir / '.completed'
    if completed_detector.is_file():
        logging.info(f'Skipping {zip_name} because {completed_detector} exists.')
        return
    if force_download or not zip_path.is_file():
        urlretrieve_progress(base_url, filename=zip_path, desc=f'Downloading {zip_name}')
    
    with zipfile.ZipFile(zip_path) as zip_file:
        corpus_dir.mkdir(parents=True, exist_ok=True)
        for names in zip_file.namelist():
            zip_file.extract(names, str(corpus_dir))


def prepare_timit(
        corpus_dir: Pathlike,
        splits_dir: Pathlike,
        output_dir: Optional[Pathlike] = None,
        num_phones: int = 48,
        num_jobs: int = 1
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consists of the Recodings and Supervisions.
    :param corpus_dir: Pathlike, the path of the data dir. 
    :param splits_dir: Pathlike, the path of the txt files for data division (train, dev, tst).
    :param output_dir: Pathlike, the path where to write and save the manifests.
    :param num_phones: int=48, the number of phones (60, 48 or 39) for modeling and 48 is regarded as the default value.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'audio' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f'No such directory: {corpus_dir}'

    splits_dir = Path(splits_dir)
    assert splits_dir.is_dir(), f'No such directory: {splits_dir}'

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    manifests = defaultdict(dict)
    dataset_parts = ['TRAIN', 'DEV', 'TEST']
    
    phones_dict = {}

    if num_phones in [60, 48, 39]:
        phones_dict = get_phonemes(num_phones)
    else:
        raise ValueError("The value of num_phones must be in [60, 48, 39].")

    with ThreadPoolExecutor(num_jobs) as ex:
        for part in dataset_parts:
            file_name = ''

            if part == 'TRAIN': 
                file_name = splits_dir / 'train_samples.txt'
            elif part == 'DEV':
                file_name = splits_dir / 'dev_samples.txt'
            else:
                file_name = splits_dir / 'tst_samples.txt'
            wav_files = []
            with open(file_name, 'r') as f:
                lines = f.readlines() 
                for line in lines:
                    items = line.strip().split(' ')
                    wav = corpus_dir / items[-1]
                    wav_files.append(wav)
                logging.debug(f'{part} dataset manifest generation.')
                recordings = []
                supervisions = []

                for wav_file in tqdm(wav_files):
                    items = str(wav_file).strip().split('/')
                    idx = items[-2] + '-' + items[-1][:-4]
                    speaker = items[-2] 
                    transcript_file = Path(wav_file).with_suffix('.PHN')
                    if not Path(wav_file).is_file():
                        logging.warning(f'No such file: {wav_file}')
                        continue
                    if not Path(transcript_file).is_file():
                        logging.warning(f'No transcript: {transcript_file}')
                        continue
                    text = []
                    with open(transcript_file, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            phone = line.rstrip('\n').split(' ')[-1]
                            if num_phones != 60: phone = phones_dict[str(phone)]
                            text.append(phone)

                        text = ' '.join(text).replace('h#', 'sil')

                    recording = Recording.from_file(path=wav_file, recording_id=idx)
                    recordings.append(recording)
                    segment = SupervisionSegment(
                        id=idx,
                        recording_id=idx,
                        start=0.0,
                        duration=recording.duration,
                        channel=0,
                        language='English',
                        speaker=speaker,
                        text=text.strip())

                    supervisions.append(segment)

                    recording_set = RecordingSet.from_recordings(recordings)
                    supervision_set = SupervisionSet.from_segments(supervisions)
                    validate_recordings_and_supervisions(recording_set, supervision_set)

                    if output_dir is not None:
                        supervision_set.to_json(output_dir / f'supervisions_{part}.json')
                        recording_set.to_json(output_dir / f'recordings_{part}.json')

                    manifests[part] = {
                        'recordings': recording_set,
                        'supervisions': supervision_set}

    return manifests

def get_phonemes(num_phones):
    """
    Choose and convert the phones for modeling.
    :param num_phones: the number of phones for modeling. 
    """
    phonemes = {}
   
    if num_phones == int(48):
        logging.debug("Using 48 phones for modeling!")
        # This dictionary is used to convert the 60 phoneme set into the 48 one.
        phonemes["sil"] = "sil"
        phonemes["aa"] = "aa"
        phonemes["ae"] = "ae"
        phonemes["ah"] = "ah"
        phonemes["ao"] = "ao"
        phonemes["aw"] = "aw"
        phonemes["ax"] = "ax"
        phonemes["ax-h"] = "ax"
        phonemes["axr"] = "er"
        phonemes["ay"] = "ay"
        phonemes["b"] = "b"
        phonemes["bcl"] = "vcl"
        phonemes["ch"] = "ch"
        phonemes["d"] = "d"
        phonemes["dcl"] = "vcl"
        phonemes["dh"] = "dh"
        phonemes["dx"] = "dx"
        phonemes["eh"] = "eh"
        phonemes["el"] = "el"
        phonemes["em"] = "m"
        phonemes["en"] = "en"
        phonemes["eng"] = "ng"
        phonemes["epi"] = "epi"
        phonemes["er"] = "er"
        phonemes["ey"] = "ey"
        phonemes["f"] = "f"
        phonemes["g"] = "g"
        phonemes["gcl"] = "vcl"
        phonemes["h#"] = "sil"
        phonemes["hh"] = "hh"
        phonemes["hv"] = "hh"
        phonemes["ih"] = "ih"
        phonemes["ix"] = "ix"
        phonemes["iy"] = "iy"
        phonemes["jh"] = "jh"
        phonemes["k"] = "k"
        phonemes["kcl"] = "cl"
        phonemes["l"] = "l"
        phonemes["m"] = "m"
        phonemes["n"] = "n"
        phonemes["ng"] = "ng"
        phonemes["nx"] = "n"
        phonemes["ow"] = "ow"
        phonemes["oy"] = "oy"
        phonemes["p"] = "p"
        phonemes["pau"] = "sil"
        phonemes["pcl"] = "cl"
        phonemes["q"] = ""
        phonemes["r"] = "r"
        phonemes["s"] = "s"
        phonemes["sh"] = "sh"
        phonemes["t"] = "t"
        phonemes["tcl"] = "cl"
        phonemes["th"] = "th"
        phonemes["uh"] = "uh"
        phonemes["uw"] = "uw"
        phonemes["ux"] = "uw"
        phonemes["v"] = "v"
        phonemes["w"] = "w"
        phonemes["y"] = "y"
        phonemes["z"] = "z"
        phonemes["zh"] = "zh"

    elif num_phones == int(39):
        logging.debug("Using 39 phones for modeling!")
        # This dictionary is used to convert the 60 phoneme set into the 39 one.
        phonemes["sil"] = "sil"
        phonemes["aa"] = "aa"
        phonemes["ae"] = "ae"
        phonemes["ah"] = "ah"
        phonemes["ao"] = "aa"
        phonemes["aw"] = "aw"
        phonemes["ax"] = "ah"
        phonemes["ax-h"] = "ah"
        phonemes["axr"] = "er"
        phonemes["ay"] = "ay"
        phonemes["b"] = "b"
        phonemes["bcl"] = "sil"
        phonemes["ch"] = "ch"
        phonemes["d"] = "d"
        phonemes["dcl"] = "sil"
        phonemes["dh"] = "dh"
        phonemes["dx"] = "dx"
        phonemes["eh"] = "eh"
        phonemes["el"] = "l"
        phonemes["em"] = "m"
        phonemes["en"] = "n"
        phonemes["eng"] = "ng"
        phonemes["epi"] = "sil"
        phonemes["er"] = "er"
        phonemes["ey"] = "ey"
        phonemes["f"] = "f"
        phonemes["g"] = "g"
        phonemes["gcl"] = "sil"
        phonemes["h#"] = "sil"
        phonemes["hh"] = "hh"
        phonemes["hv"] = "hh"
        phonemes["ih"] = "ih"
        phonemes["ix"] = "ih"
        phonemes["iy"] = "iy"
        phonemes["jh"] = "jh"
        phonemes["k"] = "k"
        phonemes["kcl"] = "sil"
        phonemes["l"] = "l"
        phonemes["m"] = "m"
        phonemes["ng"] = "ng"
        phonemes["n"] = "n"
        phonemes["nx"] = "n"
        phonemes["ow"] = "ow"
        phonemes["oy"] = "oy"
        phonemes["p"] = "p"
        phonemes["pau"] = "sil"
        phonemes["pcl"] = "sil"
        phonemes["q"] = ""
        phonemes["r"] = "r"
        phonemes["s"] = "s"
        phonemes["sh"] = "sh"
        phonemes["t"] = "t"
        phonemes["tcl"] = "sil"
        phonemes["th"] = "th"
        phonemes["uh"] = "uh"
        phonemes["uw"] = "uw"
        phonemes["ux"] = "uw"
        phonemes["v"] = "v"
        phonemes["w"] = "w"
        phonemes["y"] = "y"
        phonemes["z"] = "z"
        phonemes["zh"] = "sh"
    
    else:
        logging.debug("Using 60 phones for modeling!")

    return phonemes
