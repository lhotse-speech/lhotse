import logging
import re
import shutil
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path, PurePath
from typing import Dict, NamedTuple, Optional, Union

import torchaudio

from lhotse.audio import AudioSource, Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike

heroico_dataset_answers = ('heroico-answers.txt')
heroico_dataset_recordings = ('heroico-recordings.txt')
usma_dataset = ('usma-prompts.txt')


def download_and_untar(
        target_dir: Pathlike = '.',
        force_download: Optional[bool] = False,
        url: Optional[str] = 'http://www.openslr.org/resources/39'
) -> None:
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    tar_name = f'LDC2006S37.tar.gz'
    tar_path = target_dir / tar_name
    if force_download or not tar_path.is_file():
        urllib.request.urlretrieve(f'{url}/{tar_name}', filename=tar_path)

    completed_detector = target_dir / '.completed'
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=target_dir)
        completed_detector.touch()


class HeroicoMetaData(NamedTuple):
    audio_path: Pathlike
    audio_info: torchaudio.sox_signalinfo_t
    text: str


def prepare_heroico_answers(
        speech_dir: Pathlike,
        transcript_dir: Pathlike,
        output_dir: Optional[Pathlike] = None
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions

    :param speech_dir: Pathlike, the path of the speech data dir.
param transcripts_dir: Pathlike, the path of the transcript data dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'audio' and 'supervisions'.
    """
    speech_dir = Path(speech_dir)
    transcript_dir = Path(transcript_dir)
    assert speech_dir.is_dir(), f'No such directory: {speech_dir}'
    assert transcript_dir.is_dir(), f'No such directory: {transcript_dir}'
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    manifests = defaultdict(dict)
    line_pattern = re.compile("\d+/\d+\t.+")
    # Generate a mapping: utt_id -> (audio_path, audio_info, text)
    metadata = {}
    trans_path = transcript_dir / 'heroico-answers.txt'
    with open(trans_path, encoding='iso-8859-1') as f:
        for line in f:
            # some recordings do not have a transcript, skip them here
            if not line_pattern.match(line):
                continue
            line = line.rstrip()
            spk_utt, text = line.split(maxsplit=1)
            audio_path = speech_dir / f'{spk_utt}.wav'
            if audio_path.is_file():
                # info[0]: info of the raw audio (e.g. channel number, sample rate, duration ... )
                # info[1]: info about the encoding (e.g. FLAC/ALAW/ULAW ...)
                info = torchaudio.info(str(audio_path))
                path_components = PurePath(str(audio_path))
                prompt_id = audio_path.stem
                path_parts = path_components.parts
                speaker = path_parts[10]
                utt_id = '-'.join([path_parts[8], path_parts[9], speaker, prompt_id])
                metadata[utt_id] = HeroicoMetaData(audio_path=audio_path, audio_info=info[0], text=text)
            else:
                logging.warning(f'No such file: {audio_path}')

        # Audio
        audio = RecordingSet.from_recordings(
            Recording(
                id=utt_id,
                sources=[
                    AudioSource(
                        type='file',
                        channels=[0],
                        source=str(metadata[utt_id].audio_path)
                    )
                ],
                sampling_rate=int(metadata[utt_id].audio_info.rate),
                num_samples=metadata[utt_id].audio_info.length,
                duration=(metadata[utt_id].audio_info.length / metadata[utt_id].audio_info.rate)
            )
            for utt_id in metadata
        )

        # Supervision
        supervision = SupervisionSet.from_segments(
            SupervisionSegment(
                id=utt_id,
                recording_id=utt_id,
                start=0.0,
                duration=audio.recordings[utt_id].duration,
                channel=0,
                language='Spanish',
                speaker=utt_id.split('-')[-2],
                text=metadata[utt_id].text.strip()
            )
            for utt_id in audio.recordings
        )

        if output_dir is not None:
            supervision.to_json(output_dir / f'supervisions-heroico-answers.json')
            audio.to_json(output_dir / f'recordings-heroico-answers.json')

        manifests['answers'] = {
            'recordings': audio,
            'supervisions': supervision
        }

    return manifests

def prepare_heroico_recitations(
        speech_dir: Pathlike,
        transcript_dir: Pathlike,
        output_dir: Optional[Pathlike] = None
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions

    :param speech_dir: Pathlike, the path of the speech data dir.
param transcripts_dir: Pathlike, the path of the transcript data directory.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'audio' and 'supervisions'.
    """
    speech_dir = Path(speech_dir)
    transcript_dir = Path(transcript_dir)
    assert speech_dir.is_dir(), f'No such directory: {speech_dir}'
    assert transcript_dir.is_dir(), f'No such directory: {transcript_dir}'
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    manifests = defaultdict(dict)
    transcripts = defaultdict(dict)
    # Generate a mapping: utt_id -> (audio_path, audio_info, text)
    metadata = {}
    line_pattern = re.compile("\d+\t.+")
    audio_paths = speech_dir.rglob('*.wav')
    trans_path = Path(transcript_dir, 'heroico-recordings.txt')
    with open(trans_path, encoding='iso-8859-1') as f:
        for line in f:
            line = line.rstrip()
            if not line_pattern.match(line):
                continue
            idx, text = line.split(maxsplit=1)
            transcripts[idx] = (text)
    for wav_file in audio_paths:
        path_components = PurePath(wav_file)
        speaker = path_components.parts[-2]
        prompt_id = path_components.stem
        utt_id = '-'.join(['heroico-recitations', speaker, prompt_id])
        if wav_file.is_file():
            # info[0]: info of the raw audio (e.g. channel number, sample rate, duration ... )
            # info[1]: info about the encoding (e.g. FLAC/ALAW/ULAW ...)
            info = torchaudio.info(str(wav_file))
            metadata[utt_id] = HeroicoMetaData(audio_path=wav_file, audio_info=info[0], text=transcripts[idx])
        else:
            logging.warning(f'No such file: {audio_path, idx}')
            exit()

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
                language='Spanish',
                speaker=re.sub(r'-.*', r'', idx),
                text=metadata[idx].text.strip()
            )
            for idx in audio.recordings
        )

    if output_dir is not None:
        supervision.to_json(output_dir / f'supervisions-heroico-recitations.json')
        audio.to_json(output_dir / f'recordings-heroico-recitations.json')


    manifests['recitations'] = {
        'recordings': audio,
        'supervisions': supervision
    }

    return manifests

def prepare_usma(
        speech_dir: Pathlike,
        transcript_dir: Pathlike,
        output_dir: Optional[Pathlike] = None
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions

    :param speech_dir: Pathlike, the path of the speech data dir.
param transcripts_dir: Pathlike, the path of the transcript data directory.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'audio' and 'supervisions'.
    """
    speech_dir = Path(speech_dir)
    transcript_dir = Path(transcript_dir)
    assert speech_dir.is_dir(), f'No such directory: {speech_dir}'
    assert transcript_dir.is_dir(), f'No such directory: {transcript_dir}'
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    manifests = defaultdict(dict)
    transcripts = defaultdict(dict)
    # Generate a mapping: utt_id -> (audio_path, audio_info, text)
    metadata = {}
    line_pattern = re.compile("s\d+\t.+")
    audio_paths = speech_dir.rglob('*.wav')
    trans_path = Path(transcript_dir, usma_dataset)
    with open(trans_path, encoding='iso-8859-1') as f:
        for line in f:
            line = line.rstrip()
            if not line_pattern.match(line):
                continue
            idx, text = line.split(maxsplit=1)
            transcripts[idx] = (text)
            for wav_file in audio_paths:
                path_components = PurePath(wav_file)
                speaker = path_components.parts[-2]
                prompt_id = path_components.stem
                utt_id = '-'.join(['usma', 'recitations', speaker, prompt_id])
                if wav_file.is_file():
                    # info[0]: info of the raw audio (e.g. channel number, sample rate, duration ... )
                    # info[1]: info about the encoding (e.g. FLAC/ALAW/ULAW ...)
                    info = torchaudio.info(str(wav_file))
                    metadata[utt_id] = HeroicoMetaData(audio_path=wav_file, audio_info=info[0], text=transcripts[idx])
                else:
                    logging.warning(f'No such file: {audio_path, idx}')

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
                speaker=re.sub(r'-.*', r'', idx),
                text=metadata[idx].text.strip()
            )
            for idx in audio.recordings
        )

    if output_dir is not None:
        supervision.to_json(output_dir / f'supervisions-usma-recitations.json')
        audio.to_json(output_dir / f'recordings-usma-recitations.json')

    exit()
    manifests[usma_dataset] = {
        'recordings': audio,
        'supervisions': supervision
    }

    return manifests
