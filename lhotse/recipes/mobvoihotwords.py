"""
About the MobvoiHotwords corpus

    The MobvoiHotwords dataset is a ~144-hour corpus of wake word corpus which is
    publicly availble on https://www.openslr.org/87

    For wake word data, wake word utterances contain either 'Hi xiaowen' or 'Nihao
    Wenwen' are collected. For each wake word, there are about 36k utterances. All
    wake word data is collected from 788 subjects, ages 3-65, with different
    distances from the smart speaker (1, 3 and 5 meters). Different noises
    (typical home environment noises like music and TV) with varying sound
    pressure levels are played in the background during the collection.
"""

import json
import logging
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, NamedTuple, Optional, Union

import torchaudio

from lhotse.audio import AudioSource, Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, urlretrieve_progress


def download_and_untar(
        target_dir: Pathlike = '.',
        force_download: Optional[bool] = False,
        base_url: Optional[str] = 'http://www.openslr.org/resources'
) -> None:
    """
    Downdload and untar the dataset

    :param target_dir: Pathlike, the path of the dir to storage the dataset.
    :param force_download: Bool, if True, download the tars no matter if the tars exist.
    :param base_url: str, the url of the OpenSLR resources.
    """

    url = f'{base_url}/87'
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    dataset_tar_name = 'mobvoi_hotword_dataset.tgz'
    resources_tar_name = 'mobvoi_hotword_dataset_resources.tgz'
    for tar_name in [dataset_tar_name, resources_tar_name]:
        tar_path = target_dir / tar_name
        if force_download or not tar_path.is_file():
            urlretrieve_progress(f'{url}/{tar_name}', filename=tar_path, desc=f'Downloading {tar_name}')
        corpus_dir = target_dir / 'MobvoiHotwords'
        extracted_dir = corpus_dir / tar_name[: -4]
        completed_detector = extracted_dir / '.completed'
        if not completed_detector.is_file():
            shutil.rmtree(extracted_dir, ignore_errors=True)
            with tarfile.open(tar_path) as tar:
                tar.extractall(path=corpus_dir)
                completed_detector.touch()


class MobvoiHotwordsMetaData(NamedTuple):
    audio_path: Pathlike
    audio_info: torchaudio.sox_signalinfo_t
    speaker: str
    text: str


def prepare_mobvoihotwords(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions

    :param corpus_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'audio' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f'No such directory: {corpus_dir}'
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    manifests = defaultdict(dict)
    dataset_parts = ['train', 'dev', 'test']
    for part in dataset_parts:
        # Generate a mapping: utt_id -> (audio_path, audio_info, speaker, text)
        metadata = {}
        for prefix in ['p_', 'n_']:
            prefixed_part = prefix + part
            json_path = corpus_dir / 'mobvoi_hotword_dataset_resources' / f'{prefixed_part}.json'
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                for entry in json_data:
                    idx = entry['utt_id']
                    speaker = idx if entry['speaker_id'] is None else entry['speaker_id']
                    audio_path = corpus_dir / 'mobvoi_hotword_dataset' / f'{idx}.wav'
                    text = 'FREETEXT'
                    if entry['keyword_id'] == 0:
                        text = 'HiXiaowen'
                    elif entry['keyword_id'] == 1:
                        text = 'NihaoWenwen'
                    else:
                        assert entry['keyword_id'] == -1
                    if audio_path.is_file():
                        # info[0]: info of the raw audio (e.g. channel number, sample rate, duration ... )
                        # info[1]: info about the encoding (e.g. FLAC/ALAW/ULAW ...)
                        info = torchaudio.info(str(audio_path))
                        metadata[idx] = MobvoiHotwordsMetaData(
                            audio_path=audio_path, audio_info=info[0], speaker=speaker, text=text
                        )
                    else:
                        logging.warning(f'No such file: {audio_path}')

        # Audio
        audio = RecordingSet.from_recordings(
            Recording(
                id=idx,
                sources=[
                    AudioSource(
                        type='file',
                        channels=[0],
                        source=str(metadata[idx].audio_path)
                    )
                ],
                sampling_rate=int(metadata[idx].audio_info.rate),
                num_samples=metadata[idx].audio_info.length,
                duration=(metadata[idx].audio_info.length / metadata[idx].audio_info.rate)
            )
            for idx in metadata
        )

        # Supervision
        supervision = SupervisionSet.from_segments(
            SupervisionSegment(
                id=idx,
                recording_id=idx,
                start=0.0,
                duration=audio.recordings[idx].duration,
                channel=0,
                language='Chinese',
                speaker=metadata[idx].speaker,
                text=metadata[idx].text.strip()
            )
            for idx in audio.recordings
        )

        if output_dir is not None:
            supervision.to_json(output_dir / f'supervisions_{part}.json')
            audio.to_json(output_dir / f'recordings_{part}.json')

        manifests[part] = {
            'recordings': audio,
            'supervisions': supervision
        }

    return manifests
