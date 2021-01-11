"""
The data prepare recipe for the AMI Meeting Corpus diarization task.

In this recipe, the IHM-Mix and SDM settings are supported.

We use the Full Partition (used in Kaldi s5c recipe) which has more dev and eval sessions than the ASR partition, 
and references are obtained from the ami_manual_annotations_v1.6.2. We get both the split and the references
from https://github.com/BUTSpeechFIT/AMI-diarization-setup
"""

from collections import defaultdict

import logging
import os
import re
import soundfile
import urllib.request
from pathlib import Path

from typing import Dict, List, NamedTuple, Optional, Union

from lhotse.qa import validate_recordings_and_supervisions
from lhotse.audio import AudioSource, Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, Seconds

from lhotse.recipes.ami.common import download_audio, prepare_audio_other, remove_supervision_exceeding_audio_duration

# TODO: support the "ihm" and "mdm" microphone settings
mics = ['ihm-mix','sdm']

# Obtain data split from BUT-diarization-setup
dataset_parts = {}
base_url="https://raw.githubusercontent.com/BUTSpeechFIT/AMI-diarization-setup/main/lists/"

for split in ['train','dev','test']:
    dataset_parts[split] = []
    target_url = f"{base_url}{split}.meetings.txt"
    for line in urllib.request.urlopen(target_url):
        line = line.decode("utf-8")
        dataset_parts[split].append(line.strip())


def download(
        target_dir: Pathlike = '.',
        force_download: Optional[bool] = False,
        url: Optional[str] = 'http://groups.inf.ed.ac.uk/ami',
        mic: Optional[str] = 'ihm-mix'
) -> None:

    assert mic in mics, "Only {mics} mics are supported for now."
    target_dir = Path(target_dir)
    # Audio
    download_audio(target_dir, dataset_parts, force_download, url, mic)

    # Annotations/References
    rttm_name = 'rttms'
    rttm_path = target_dir / rttm_name
    base_url = 'https://raw.githubusercontent.com/BUTSpeechFIT/AMI-diarization-setup/main/only_words/rttms/{}/{}.rttm'
    if force_download or not rttm_path.is_dir():
        rttm_path.mkdir(parents=True, exist_ok=True)
        for split in dataset_parts:
            for session in dataset_parts[split]:
                cur_rttm_path = rttm_path / f'{session}.rttm'
                rttm_url = base_url.format(split, session)
                urllib.request.urlretrieve(rttm_url, filename=cur_rttm_path)


class AmiSegmentAnnotation(NamedTuple):
    speaker: str
    gender: str
    begin_time: Seconds
    end_time: Seconds

def parse_ami_annotations(
    rttm_path: Pathlike,
) -> Dict[str, List[AmiSegmentAnnotation]]:
    annotations = {}
    for file in os.listdir(rttm_path):
        meet_id = file.split('.')[0]
        filepath = rttm_path / file
        annotations[meet_id] = []
        with open(filepath, 'r') as f:
            for line in f:
                _, _, _, start, dur, _, _, spk, _, _ = line.strip().split()
                annotations[meet_id].append(AmiSegmentAnnotation(
                    speaker=spk,
                    gender=spk[0],
                    begin_time=float(start),
                    end_time=float(start)+float(dur)
                ))
    return annotations

def prepare_supervision(
    audio: Dict[str, RecordingSet],
    annotations: Dict[str, List[AmiSegmentAnnotation]]
) -> Dict[str, SupervisionSet]:
    supervision_manifest = defaultdict(dict)

    for part in dataset_parts:
        segments = []
        audio_part = audio[part]
        for recording in audio_part:
            annotation = annotations.get(recording.id)
            # In IHM-Mix and SDM, there can only be 1 source per recording, but it
            # can sometimes have more than 1 channels. But we only care about
            # 1 channel so we only add supervision for that channel in the
            # supervision manifest.
            source, = recording.sources
            if annotation is None:
                logging.warning(f'No annotation found for recording {recording.id} '
                                f'(file {source.source})')
                continue
            
            if (len(source.channels) > 1):
                logging.warning(f'More than 1 channels in recording {recording.id}. '
                                f'Creating supervision for channel 0 only.')

            for seg_idx, seg_info in enumerate(annotation):
                duration = seg_info.end_time - seg_info.begin_time
                if duration > 0:
                    segments.append(SupervisionSegment(
                        id=f'{recording.id}-{seg_idx}',
                        recording_id=recording.id,
                        start=seg_info.begin_time,
                        duration=duration,
                        channel=0,
                        language='English',
                        speaker=seg_info.speaker,
                        gender=seg_info.gender
                    ))

        supervision_manifest[part] = SupervisionSet.from_segments(segments)
    return supervision_manifest

def prepare_ami(
        data_dir: Pathlike,
        output_dir: Optional[Pathlike] = None,
        mic: Optional[str] = 'ihm-mix'
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param data_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param mic: str, type of mic to use.
    :return: a Dict whose key is ('train', 'dev', 'eval'), and the value is Dicts with keys 'audio' and 'supervisions'.
    """
    data_dir = Path(data_dir)
    assert data_dir.is_dir(), f'No such directory: {data_dir}'
    assert mic in mics, f'Mic {mic} not supported'

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Audio
    wav_dir = data_dir / 'wav_db'
    if mic == 'ihm-mix':
        audio_paths = wav_dir.rglob('*Mix-Headset.wav')
        audio = prepare_audio_other(list(audio_paths), dataset_parts)
    elif mic == 'sdm':
        audio_paths = wav_dir.rglob('*Array1-01.wav')
        audio = prepare_audio_other(list(audio_paths), dataset_parts)
    
    annotations = parse_ami_annotations(data_dir / 'rttms')
    supervision = prepare_supervision(audio, annotations)

    manifests = defaultdict(dict)

    for part in dataset_parts:

        # Write to output directory if a path is provided
        if output_dir is not None:
            audio[part].to_json(output_dir / f'recordings_{part}.json')
            supervision[part].to_json(output_dir / f'supervisions_{part}.json')

        # NOTE: Some of the AMI annotations exceed the recording duration, so
        # we remove such segments here
        supervision[part] = remove_supervision_exceeding_audio_duration(
            audio[part],
            supervision[part],
        )
        validate_recordings_and_supervisions(audio[part], supervision[part])
        
        # Combine all manifests into one dictionary
        manifests[part] = {
        'recordings': audio[part],
        'supervisions': supervision[part]
        }

    return manifests
