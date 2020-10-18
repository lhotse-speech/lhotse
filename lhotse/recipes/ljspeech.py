"""
The LJ Speech Dataset is a public domain speech dataset consisting of 13,100 short audio clips of a single speaker
reading passages from 7 non-fiction books. A transcription is provided for each clip. Clips vary in length from 1 to
10 seconds and have a total length of approximately 24 hours.

The texts were published between 1884 and 1964, and are in the public domain. The audio was recorded in 2016-17 by
the LibriVox project and is also in the public domain.

See https://keithito.com/LJ-Speech-Dataset for more details.
"""

import logging
import re
import shutil
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, NamedTuple, Optional, Union

import torchaudio

from lhotse.audio import AudioSource, Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.features.base import TorchaudioFeatureExtractor
from lhotse.features import Fbank
from lhotse.utils import Pathlike, fastcopy


def download_and_untar(
        target_dir: Pathlike = '.',
        force_download: Optional[bool] = False
) -> None:
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = 'LJSpeech-1.1'
    tar_path = target_dir / f'{dataset_name}.tar.bz2'
    if force_download or not tar_path.is_file():
        urllib.request.urlretrieve(f'http://data.keithito.com/data/speech/{dataset_name}.tar.bz2', filename=tar_path)
    corpus_dir = target_dir / dataset_name
    completed_detector = corpus_dir / '.completed'
    if not completed_detector.is_file():
        shutil.rmtree(corpus_dir, ignore_errors=True)
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=target_dir)
            completed_detector.touch()


class LJSpeechMetaData(NamedTuple):
    audio_path: Pathlike
    audio_info: torchaudio.sox_signalinfo_t
    text: str


def prepare_ljspeech(
        corpus_dir: Pathlike,
        output_dir: Optional[Pathlike] = None
) -> Dict[str, Union[RecordingSet, SupervisionSet]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions

    :param corpus_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: The RecordingSet and SupervisionSet with the keys 'audio' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f'No such directory: {corpus_dir}'
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Generate a mapping: utt_id -> (audio_path, audio_info, text)
    metadata_csv_path = corpus_dir / 'metadata.csv'
    assert metadata_csv_path.is_file(), f'No such file: {metadata_csv_path}'
    metadata = {}
    with open(metadata_csv_path) as f:
        for line in f:
            idx, text, _ = line.split('|')
            audio_path = corpus_dir / 'wavs' / f'{idx}.wav'
            if audio_path.is_file():
                # info[0]: info of the raw audio (e.g. channel number, sample rate, duration ... )
                # info[1]: info about the encoding (e.g. FLAC/ALAW/ULAW ...)
                info = torchaudio.info(str(audio_path))
                metadata[idx] = LJSpeechMetaData(audio_path=audio_path, audio_info=info[0], text=text)
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
            language='English',
            gender='female',
            text=metadata[idx].text
        )
        for idx in audio.recordings
    )

    if output_dir is not None:
        supervision.to_json(output_dir / 'supervisions.json')
        audio.to_json(output_dir / 'audio.json')

    return {'audio': audio, 'supervisions': supervision}


def feature_extractor() -> TorchaudioFeatureExtractor:
    """
    Set up the feature extractor for TTS task.
    :return: A feature extractor with custom parameters.
    """
    feature_extractor = Fbank()
    feature_extractor.config.num_mel_bins = 80

    return feature_extractor


def text_normalizer(segment: SupervisionSegment) -> SupervisionSegment:
    text = segment.text.upper()
    text = re.sub(r'[^\w !?]', '', text)
    text = re.sub(r'^\s+', '', text)
    text = re.sub(r'\s+$', '', text)
    text = re.sub(r'\s+', ' ', text)
    return fastcopy(segment, text=text)
