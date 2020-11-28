"""
The data prepare recipe for the AMI Meeting Corpus.

The AMI Meeting Corpus consists of 100 hours of meeting recordings. The recordings use a range of signals
synchronized to a common timeline. These include close-talking and far-field microphones, individual and room-view
video cameras, and output from a slide projector and an electronic whiteboard. During the meetings, the participants
also have unsynchronized pens available to them that record what is written. The meetings were recorded in English
using three different rooms with different acoustic properties, and include mostly non-native speakers." See
http://groups.inf.ed.ac.uk/ami/corpus/overview.shtml for more details.

There are several microphone settings in AMI corpus:
-- IHM: Individual Headset Microphones
-- SDM: Single Distant Microphone
-- MDM: Multiple Distant Microphones

In this recipe, IHM, IHM-Mix, and SDM settings are supported.
"""

import logging
import os
import re
import urllib.request
import zipfile
from collections import defaultdict
import xml.etree.ElementTree as ET
from gzip import GzipFile
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Union, Tuple

# Workaround for SoundFile (torchaudio dep) raising exception when a native library, libsndfile1, is not installed.
# Read-the-docs does not allow to modify the Docker containers used to build documentation...
from tqdm.auto import tqdm
import soundfile as sf

from lhotse.audio import AudioSource, Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, Seconds

# TODO: support the "mdm" microphone setting. For "sdm" we currently use Array1-01.
mics = ['ihm','ihm-mix','sdm']

# The splits are same with Kaldi's AMI recipe
split_train = (
    'EN2001a', 'EN2001b', 'EN2001d', 'EN2001e', 'EN2003a', 'EN2004a', 'EN2005a', 'EN2006a', 'EN2006b', 'EN2009b',
    'EN2009c', 'EN2009d', 'ES2002a', 'ES2002b', 'ES2002c', 'ES2002d', 'ES2003a', 'ES2003b', 'ES2003c', 'ES2003d',
    'ES2005a', 'ES2005b', 'ES2005c', 'ES2005d', 'ES2006a', 'ES2006b', 'ES2006c', 'ES2006d', 'ES2007a', 'ES2007b',
    'ES2007c', 'ES2007d', 'ES2008a', 'ES2008b', 'ES2008c', 'ES2008d', 'ES2009a', 'ES2009b', 'ES2009c', 'ES2009d',
    'ES2010a', 'ES2010b', 'ES2010c', 'ES2010d', 'ES2012a', 'ES2012b', 'ES2012c', 'ES2012d', 'ES2013a', 'ES2013b',
    'ES2013c', 'ES2013d', 'ES2014a', 'ES2014b', 'ES2014c', 'ES2014d', 'ES2015a', 'ES2015b', 'ES2015c', 'ES2015d',
    'ES2016a', 'ES2016b', 'ES2016c', 'ES2016d', 'IB4005', 'IN1001', 'IN1002', 'IN1005', 'IN1007', 'IN1008',
    'IN1009', 'IN1012', 'IN1013', 'IN1014', 'IN1016', 'IS1000a', 'IS1000b', 'IS1000c', 'IS1000d', 'IS1001a',
    'IS1001b', 'IS1001c', 'IS1001d', 'IS1002b', 'IS1002c', 'IS1002d', 'IS1003a', 'IS1003b', 'IS1003c', 'IS1003d',
    'IS1004a', 'IS1004b', 'IS1004c', 'IS1004d', 'IS1005a', 'IS1005b', 'IS1005c', 'IS1006a', 'IS1006b', 'IS1006c',
    'IS1006d', 'IS1007a', 'IS1007b', 'IS1007c', 'IS1007d', 'TS3005a', 'TS3005b', 'TS3005c', 'TS3005d', 'TS3006a',
    'TS3006b', 'TS3006c', 'TS3006d', 'TS3007a', 'TS3007b', 'TS3007c', 'TS3007d', 'TS3008a', 'TS3008b', 'TS3008c',
    'TS3008d', 'TS3009a', 'TS3009b', 'TS3009c', 'TS3009d', 'TS3010a', 'TS3010b', 'TS3010c', 'TS3010d', 'TS3011a',
    'TS3011b', 'TS3011c', 'TS3011d', 'TS3012a', 'TS3012b', 'TS3012c', 'TS3012d')

split_dev = (
    'ES2011a', 'ES2011b', 'ES2011c', 'ES2011d', 'IB4001', 'IB4002', 'IB4003', 'IB4004', 'IB4010', 'IB4011', 'IS1008a',
    'IS1008b', 'IS1008c', 'IS1008d', 'TS3004a', 'TS3004b', 'TS3004c', 'TS3004d')

split_eval = (
    'EN2002a', 'EN2002b', 'EN2002c', 'EN2002d', 'ES2004a', 'ES2004b', 'ES2004c', 'ES2004d', 'IS1009a', 'IS1009b',
    'IS1009c', 'IS1009d', 'TS3003a', 'TS3003b', 'TS3003c', 'TS3003d')

dataset_parts = {'train': split_train, 'dev': split_dev, 'eval': split_eval}


def download(
        target_dir: Pathlike = '.',
        mic: Optional[str] = 'ihm',
        force_download: Optional[bool] = False,
        url: Optional[str] = 'http://groups.inf.ed.ac.uk/ami',
        alignments: Optional[bool] = False,
) -> None:
    assert mic in mics
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

    # Annotations
    text_name = 'annotations.gzip'
    text_path = target_dir / text_name
    if force_download or not text_path.is_file():
        text_url = f'{url}/AMICorpusAnnotations/ami_manual_annotations_v1.6.1_export.gzip'
        urllib.request.urlretrieve(text_url, filename=text_path)

    if alignments:
        fa_name = 'word_annotations.zip'
        fa_path = target_dir / fa_name
        fa_url = f'{url}/AMICorpusAnnotations/ami_public_manual_1.6.1.zip'
        if force_download or not fa_path.is_file():
            urllib.request.urlretrieve(fa_url, filename=fa_path)


class AmiSegmentAnnotation(NamedTuple):
    text: str
    speaker: str
    gender: str
    begin_time: Seconds
    end_time: Seconds


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


def parse_ami_annotations(gzip_file: Pathlike, mic: Optional[str]='ihm'
    ) -> Dict[str, List[AmiSegmentAnnotation]]:
    annotations = {}
    speaker_dict = {}
    channel_dict = {}

    with GzipFile(gzip_file) as f:
        for line in f:
            line = line.decode()
            line = re.sub(r'\s+$', '', line)
            line = re.sub(r'\.\.+', '', line)
            if re.match(r'^Found', line) or re.match(r'^[Oo]bs', line):
                continue
            meet_id, x, speaker, channel, transcriber_start, transcriber_end, starttime, endtime, trans, puncts_times \
                = line.split('\t')
            speaker_dict[(meet_id,x)] = speaker
            channel_dict[(meet_id,x)] = channel
            # Cleanup the transcript and split by punctuations
            trans = trans.upper()
            trans = re.sub(r' *[.,?!:]+ *', ',', trans.upper())
            trans = re.sub(r',+', ',', trans)
            trans = re.sub(r',$', '', trans)
            trans = re.split(r',', trans)

            # Time points of punctuation marks
            puncts_times = re.sub(r'-', ' ', puncts_times)
            puncts_times = re.sub(r'NaN', ' ', puncts_times)
            puncts_times = re.sub(r' +', ' ', puncts_times)

            # Unique the puncts_times and convert its items to float
            puncts_times_seen = set()
            puncts_times = [x for x in puncts_times.split() if not (x in puncts_times_seen or puncts_times_seen.add(x))]
            puncts_times = [float(t) for t in puncts_times]

            try:
                starttime = float(starttime)
                endtime = float(endtime)
                transcriber_start = float(transcriber_start)
                transcriber_end = float(transcriber_end)

                # Haven't found out why there are both 'transcriber_start' and 'starttime' columns in the
                # ami_manual_annotations_v1.6.1_export.txt and what are the differences between them. Just choose the
                # time points for the longer segments here.
                if endtime < transcriber_end:
                    endtime = transcriber_end
                if starttime > transcriber_start:
                    starttime = transcriber_start

                # Add a full stop mark if there is no punctuations
                seg_num = len(trans)
                assert seg_num > 0
                if len(puncts_times) == seg_num:
                    # In this case, the punctuation should be at the end of the sentence, so we just pop it.
                    puncts_times.pop()

                # Catches the unmatched punctuations and those time points
                assert len(puncts_times) + 1 == seg_num

                # If a punctuation mark's time point be outside of the segment, the annotation line must be invalid.
                if len(puncts_times) > 0:
                    assert starttime <= puncts_times[0] and puncts_times[-1] <= endtime

            except (ValueError, AssertionError):
                continue

            # Make the begin/end points list
            seg_btimes = [starttime] + puncts_times
            seg_btimes.pop()
            seg_times = [AmiSegmentAnnotation(
                text=t,
                speaker=speaker,
                gender=speaker[0],
                begin_time=b,
                end_time=e
            ) for t, b, e in zip(trans, seg_btimes, puncts_times)]

            wav_name = get_wav_name(mic, meet_id, int(channel))
            if wav_name not in annotations:
                 annotations[wav_name] = []
            annotations[wav_name].append(seg_times)

    return annotations, speaker_dict, channel_dict

def parse_ami_alignments(
        alignments_zip: Pathlike,
        speaker_dict: Dict[Tuple[str, str], str],
        channel_dict: Dict[Tuple[str, str], str],
        mic: Optional[str] = 'ihm'
) -> Dict[str, List[AmiSegmentAnnotation]]:
    alignments = {}
    with zipfile.ZipFile(alignments_zip, 'r') as archive:
        for file in archive.namelist():
            if file.startswith('words/') and file[-1] != '/':
                meet_id, x, _, _ = file.split('/')[1].split('.')
                # Get speaker and channel from the XML file name. Some of them
                # may not be present in the annotations (so we ignore those)
                try:
                    spk = speaker_dict[(meet_id,x)]
                    channel = channel_dict[(meet_id, x)]
                except:
                    logging.warning(f'{meet_id}.{x} present in alignments but not in '
                                    f'annotations. Removing from alignments.')
                    continue
                tree = ET.parse(archive.open(file))
                seg_times = []
                for child in tree.getroot():
                    # If the alignment does not contain start or end time info,
                    # ignore them. Also, only consider words or vocal sounds
                    # in the alignment XML files.
                    if 'starttime' not in child.attrib or 'endtime' not in child.attrib or \
                        child.tag not in ['w','vocalsound']:
                        continue
                    text = child.text if child.tag == 'w' else child.attrib['type']
                    seg_times.append(AmiSegmentAnnotation(
                        text=text,
                        speaker=spk,
                        gender=spk[0],
                        begin_time=float(child.attrib['starttime']),
                        end_time=float(child.attrib['endtime'])))
                
                wav_name = get_wav_name(mic, meet_id, channel)
                if wav_name not in alignments:
                    alignments[wav_name] = []
                alignments[wav_name].append(seg_times)
    return alignments

# Since IHM audio requires grouping multiple channels of AudioSource into
# one Recording, we separate it from preparation of SDM and IHM-mix, since
# they do not require such grouping.
def prepare_audio_ihm(
        audio_paths: List[Pathlike],
        annotations: Dict[str, List[AmiSegmentAnnotation]],
        alignments: Optional[Dict[str, List[AmiSegmentAnnotation]]] = None
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
        annotations: Dict[str, List[AmiSegmentAnnotation]],
        alignments: Optional[Dict[str, List[AmiSegmentAnnotation]]] = None
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


# Similar to audio preparation, we also need to prepare the supervisions for
# IHM differently from those for IHM-Mix and SDM mics.
def prepare_supervision_ihm(
    audio: Dict[str, RecordingSet],
    annotations: Dict[str, List[AmiSegmentAnnotation]]
) -> Dict[str, SupervisionSet]:
    # Create a mapping from a tuple of (session_id, channel) to the list of annotations.
    # This way we can map the supervisions to the right channels in a multi-channel recording.
    annotation_by_id_and_channel = {
        (filename.split('.')[0], int(filename[-5])): annot
        for filename, annot in  annotations.items()
    }
    supervision_manifest = defaultdict(dict)

    for part in dataset_parts:
        audio_part = audio[part]
        segments = []
        for recording in audio_part:
            # AMI IHM can have multiple audio sources for each recording
            for source in recording.sources:
                # For each source, "channels" will always be a one-element list
                channel, = source.channels
                annotation = annotation_by_id_and_channel.get((recording.id, channel))
                if annotation is None:
                    logging.warning(f'No annotation found for recording "{recording.id}" '
                                    f'(file {source.source})')
                    continue
                for seg_idx, seg_info in enumerate(annotation):
                    for subseg_idx, subseg_info in enumerate(seg_info):
                        duration = subseg_info.end_time - subseg_info.begin_time
                        if duration > 0:
                            segments.append(SupervisionSegment(
                                id=f'{recording.id}-{seg_idx}-{subseg_idx}',
                                recording_id=recording.id,
                                start=subseg_info.begin_time,
                                duration=duration,
                                channel=channel,
                                language='English',
                                speaker=subseg_info.speaker,
                                gender=subseg_info.gender,
                                text=subseg_info.text
                            ))

        supervision_manifest[part] = SupervisionSet.from_segments(segments)
    return supervision_manifest

def prepare_supervision_other(
    audio: Dict[str, RecordingSet],
    annotations: Dict[str, List[AmiSegmentAnnotation]]
) -> Dict[str, SupervisionSet]:
    annotation_by_id = {
        (filename.split('.')[0]): annot
        for filename, annot in  annotations.items()
    }
    supervision_manifest = defaultdict(dict)

    for part in dataset_parts:
        segments = []
        audio_part = audio[part]
        for recording in audio_part:
            annotation = annotation_by_id.get(recording.id)
            if annotation is None:
                logging.warning(f'No annotation found for recording {recording.id} '
                                f'(file {source.source})')
                continue
            # In IHM-Mix and SDM, there can only be 1 source per recording, but it
            # can sometimes have more than 1 channels. But we only care about
            # 1 channel so we only add supervision for that channel in the
            # supervision manifest.
            source, = recording.sources
            if (len(source.channels) > 1):
                logging.warning(f'More than 1 channels in recording {recording.id}. '
                                f'Creating supervision for channel 0 only.')

            for seg_idx, seg_info in enumerate(annotation):
                for subseg_idx, subseg_info in enumerate(seg_info):
                    duration = subseg_info.end_time - subseg_info.begin_time
                    if duration > 0:
                        segments.append(SupervisionSegment(
                            id=f'{recording.id}-{seg_idx}-{subseg_idx}',
                            recording_id=recording.id,
                            start=subseg_info.begin_time,
                            duration=duration,
                            channel=0,
                            language='English',
                            speaker=subseg_info.speaker,
                            gender=subseg_info.gender,
                            text=subseg_info.text
                        ))

        supervision_manifest[part] = SupervisionSet.from_segments(segments)
    return supervision_manifest

def prepare_ami(
        data_dir: Pathlike,
        output_dir: Optional[Pathlike] = None,
        mic: Optional[str] = 'ihm',
        word_alignments: Optional[Pathlike] = None
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    If word alignment archive is provided, also returns a Supervision manifest
    corresponding to alignments.

    :param data_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param mic: str, type of mic to use.
    :param word_alignments: Pathlike, the path to AMI corpus alignments archive
    :return: a Dict whose key is ('train', 'dev', 'eval'), and the value is Dicts with keys 'audio' and 'supervisions'.
    """
    data_dir = Path(data_dir)
    assert data_dir.is_dir(), f'No such directory: {data_dir}'
    assert mic in mics, f'Mic {mic} not supported'

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    annotations, speaker_dict, channel_dict = parse_ami_annotations(data_dir / 'annotations.gzip', mic=mic)
    alignments= parse_ami_alignments(word_alignments, speaker_dict, channel_dict, mic) \
        if word_alignments else None

    # Audio
    wav_dir = data_dir / 'wav_db'
    if mic == 'ihm':
        audio_paths = wav_dir.rglob('*Headset-?.wav')
        audio = prepare_audio_ihm(audio_paths, annotations, alignments)

    elif mic == 'ihm-mix':
        audio_paths = wav_dir.rglob('*Mix-Headset.wav')
        audio = prepare_audio_other(audio_paths, annotations, alignments)
    elif mic == 'sdm':
        audio_paths = wav_dir.rglob('*Array1-01.wav')
        audio = prepare_audio_other(audio_paths, annotations, alignments)

    # Supervisions
    if mic == 'ihm':
        supervision = prepare_supervision_ihm(audio, annotations)
        ali_supervision = prepare_supervision_ihm(audio, alignments) if word_alignments else None
    else:
        supervision = prepare_supervision_other(audio, annotations)
        ali_supervision = prepare_supervision_other(audio, alignments) if word_alignments else None

    manifests = defaultdict(dict)

    for part in dataset_parts:

        # Write to output directory if a path is provided
        if output_dir is not None:
            audio[part].to_json(output_dir / f'recordings_{part}.json')
            supervision[part].to_json(output_dir / f'supervisions_{part}.json')
            if word_alignments:
                ali_supervision[part].to_json(output_dir / f'alignments_{part}.json')

        # Combine all manifests into one dictionary
        manifests[part] = {
        'recordings': audio[part],
        'supervisions': supervision[part]
        }
        if word_alignments:
            manifests[part]['alignments'] = ali_supervision[part]

    return manifests
