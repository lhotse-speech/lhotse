"""
This file contains utility functions which are common to all the AMI tasks.

The AMI Meeting Corpus consists of 100 hours of meeting recordings. The recordings use a range of signals
synchronized to a common timeline. These include close-talking and far-field microphones, individual and room-view
video cameras, and output from a slide projector and an electronic whiteboard. During the meetings, the participants
also have unsynchronized pens available to them that record what is written. The meetings were recorded in English
using three different rooms with different acoustic properties, and include mostly non-native speakers." See
http://groups.inf.ed.ac.uk/ami/corpus/overview.shtml for more details.

There are several microphone settings in AMI corpus:
-- IHM: Individual Headset Microphones
-- IHM-Mix: Sum of headset microphones
-- SDM: Single Distant Microphone
-- MDM: Multiple Distant Microphones

NOTE: It is not yet clear which references and partitions should be used for joint training/evaluation
of different tasks (such as ASR+diarization), but: (i) the v1.6.1 of annotations has been found to
contain several alignment errors, and (ii) the Full Corpus ASR partition contains training speakers
in the test sets as well. 
"""
import urllib.request
import logging
import soundfile as sf

from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Optional

from tqdm.auto import tqdm

from lhotse.utils import Pathlike
from lhotse.audio import AudioSource, Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet

def download_audio(
    target_dir: Pathlike,
    dataset_parts: Dict[str,List[str]],
    force_download: Optional[bool] = False,
    url: Optional[str] = 'http://groups.inf.ed.ac.uk/ami',
    mic: Optional[str] = 'ihm'
) -> None:
    
    target_dir = Path(target_dir)

    # Audios
    for part in tqdm(dataset_parts, desc='Downloading AMI splits'):
        for item in tqdm(dataset_parts[part], desc='Downloading sessions'):
            if mic == 'ihm':
                headset_num = 5 if item in ('EN2001a', 'EN2001d', 'EN2001e') else 4
                for m in range(headset_num):
                    wav_name = f'{item}.Headset-{m}.wav'
                    wav_url = f'{url}/AMICorpusMirror/amicorpus/{item}/audio/{wav_name}'
                    wav_dir = target_dir / 'wav_db' / item / 'audio'
                    wav_dir.mkdir(parents=True, exist_ok=True)
                    wav_path = wav_dir / wav_name
                    if force_download or not wav_path.is_file():
                        urllib.request.urlretrieve(wav_url, filename=wav_path)
            elif mic == 'ihm-mix':
                wav_name = f'{item}.Mix-Headset.wav'
                wav_url = f'{url}/AMICorpusMirror/amicorpus/{item}/audio/{wav_name}'
                wav_dir = target_dir / 'wav_db' / item / 'audio'
                wav_dir.mkdir(parents=True, exist_ok=True)
                wav_path = wav_dir / wav_name
                if force_download or not wav_path.is_file():
                    urllib.request.urlretrieve(wav_url, filename=wav_path)
            elif mic == 'sdm':
                wav_name = f'{item}.Array1-01.wav'
                wav_url = f'{url}/AMICorpusMirror/amicorpus/{item}/audio/{wav_name}'
                wav_dir = target_dir / 'wav_db' / item / 'audio'
                wav_dir.mkdir(parents=True, exist_ok=True)
                wav_path = wav_dir / wav_name
                if force_download or not wav_path.is_file():
                    urllib.request.urlretrieve(wav_url, filename=wav_path)
                    
def get_wav_name(
        mic: str,
        meet_id: str,
        channel: Optional[int] = None
) -> str:
    """
    Helper function to get wav file name, given meeting id, mic, and channel
    information.
    """
    if mic == 'ihm':
        wav_name = f'{meet_id}.Headset-{channel}.wav'
    elif mic == 'ihm-mix':
        wav_name = f'{meet_id}.Mix-Headset.wav'
    elif mic == 'sdm':
        wav_name = f'{meet_id}.Array1-01.wav'
    return wav_name

# Since IHM audio requires grouping multiple channels of AudioSource into
# one Recording, we separate it from preparation of SDM and IHM-mix, since
# they do not require such grouping.
def prepare_audio_ihm(
        audio_paths: List[Pathlike],
        dataset_parts: Dict[str,List[str]],
) -> Dict[str, RecordingSet]:
    # Group together multiple channels from the same session.
    # We will use that to create a Recording with multiple sources (channels).
    from cytoolz import groupby
    channel_wavs = groupby(lambda p: p.parts[-3], audio_paths)
    recording_manifest = defaultdict(dict)

    for part in dataset_parts:
        recordings = []
        for session_name, channel_paths in channel_wavs.items():

            if session_name not in dataset_parts[part]:
                continue
            audio_sf = sf.SoundFile(str(channel_paths[0]))
            recordings.append(Recording(
                id=session_name,
                sources=[
                    AudioSource(
                        type='file',
                        channels=[idx],
                        source=str(audio_path)
                    )
                    for idx, audio_path in enumerate(sorted(channel_paths))
                ],
                sampling_rate=audio_sf.samplerate,
                num_samples=audio_sf.frames,
                duration=audio_sf.frames / audio_sf.samplerate,
            ))
        audio = RecordingSet.from_recordings(recordings)
        recording_manifest[part] = audio
    return recording_manifest


def prepare_audio_other(
        audio_paths: List[Pathlike],
        dataset_parts: Dict[str,List[str]],
) -> Dict[str, RecordingSet]:

    recording_manifest = defaultdict(dict)

    for part in dataset_parts:
        recordings = []
        for audio_path in audio_paths:
            session_name = audio_path.parts[-3]
            if session_name not in dataset_parts[part]:
                continue
            audio_sf = sf.SoundFile(str(audio_path))
            recordings.append(Recording(
                id=session_name,
                sources=[
                    AudioSource(
                        type='file',
                        channels=list(range(audio_sf.channels)),
                        source=str(audio_path)
                    )
                ],
                sampling_rate=audio_sf.samplerate,
                num_samples=audio_sf.frames,
                duration=audio_sf.frames / audio_sf.samplerate,
            ))
        audio = RecordingSet.from_recordings(recordings)
        recording_manifest[part] = audio
    return recording_manifest

def remove_supervision_exceeding_audio_duration(
    recordings: RecordingSet,
    supervision: SupervisionSet,
) -> SupervisionSet:
    return SupervisionSet.from_segments(
        s for s in supervision if s.end <= recordings[s.recording_id].duration
    )
