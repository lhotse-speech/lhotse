import os
import re
import tarfile
import urllib.request
import logging
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import Optional, Dict, Union

import torchaudio

from lhotse.audio import RecordingSet, Recording, AudioSource
from lhotse.supervision import SupervisionSet, SupervisionSegment
from lhotse.features import FeatureSet
from lhotse.cut import CutSet
from lhotse.utils import Pathlike, Seconds, find_files_in_directory

dataset_parts = ('dev-clean-2', 'train-clean-5')


def download_and_untar(
        target_path: Pathlike = '.',
        force_download: Optional[bool] = False,
        url: Optional[str] = 'http://www.openslr.org/resources/31'
) -> None:
    for part in dataset_parts:
        tar_name = f'{part}.tar.gz'
        tar_path = target_path / tar_name
        if force_download or not tar_path.is_file():
            urllib.request.urlretrieve(f'{url}/{tar_name}', filename=tar_path)
        completed_detector = target_path / f'LibriSpeech/{part}/.completed'
        if not completed_detector.is_file():
            with tarfile.open(tar_path) as tar:
                tar.extractall(path=target_path)
                completed_detector.touch()



def prepare_mini_librispeech(
        corpus_dir: Pathlike,
        output_dir: Pathlike,
        min_segment_seconds: Seconds = 3.0
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifests = defaultdict(dict)

    for part in dataset_parts:
        # Generate a mapping: utt_id -> (audio_path, audio_info, text)
        metadata = defaultdict(dict)
        MetaDataType = namedtuple('MetaType', ['audio_path', 'audio_info', 'text'])
        part_path = corpus_dir / part
        for trans_path in part_path.rglob('*.txt'):
            with open(trans_path) as f:
                for line in f:
                    idx, text = line.split(maxsplit=1)
                    audio_path = part_path / Path(idx.replace('-', '/')).parent / f'{idx}.flac'
                    if audio_path.is_file():
                        audio_path = str(audio_path)
                        metadata[idx] = MetaDataType(audio_path=audio_path, audio_info=torchaudio.info(audio_path), text=text)
                    else:
                        logging.warning('No such file: {}'.format(audio_path))

        # Audio
        audio = RecordingSet.from_recordings(
            Recording(
                id=idx,
                sources=[
                    AudioSource(
                        type='file',
                        channel_ids=[0],
                        source=metadata[idx].audio_path
                    )
                ],
                sampling_rate=int(metadata[idx].audio_info[0].rate),
                num_samples=metadata[idx].audio_info[0].length,
                duration_seconds=(metadata[idx].audio_info[0].length / metadata[idx].audio_info[0].rate)
            )
            for idx in metadata
        )
        audio.to_yaml(output_dir / 'audio_{}.yml'.format(part))

        # Supervision
        supervision = SupervisionSet.from_segments(
            SupervisionSegment(
                id=idx,
                recording_id=idx,
                start=0.0,
                duration=audio.recordings[idx].duration_seconds,
                channel_id=0,
                language='English',
                speaker=re.sub(r'-.*', r'', idx),
                text=metadata[idx].text
            )
            for idx in audio.recordings
        )
        supervision.to_yaml(output_dir / 'supervisions_{}.yml'.format(part))

        manifests[part] = {
            'audio': audio,
            'supervisions': supervision
        }

    return manifests
