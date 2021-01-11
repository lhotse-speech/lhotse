"""
The data prepare recipe for the AMI Meeting Corpus ASR task.

In this recipe, the IHM, IHM-Mix, and SDM settings are supported.

We use the Full Partition ASR corpus (used in Kaldi s5 and s5b recipes), and the references
are obtained from ami_manual_annotations_v1.6.1. This is done for the data preparation
to be compatible with the Kaldi recipe (but may have to be updated in future).
"""

from collections import defaultdict
from gzip import GzipFile

import logging
import re
import soundfile
import urllib.request
from pathlib import Path
from tqdm.auto import tqdm
from typing import Dict, List, NamedTuple, Optional, Union

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import AudioSource, Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, Seconds

from lhotse.recipes.ami.common import download_audio, get_wav_name, prepare_audio_ihm, prepare_audio_other, \
    remove_supervision_exceeding_audio_duration

# TODO: support the "mdm" microphone setting
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
        force_download: Optional[bool] = False,
        url: Optional[str] = 'http://groups.inf.ed.ac.uk/ami',
        mic: Optional[str] = 'ihm'
) -> None:
    
    assert mic in mics, "Only {mics} mics are supported for now."
    
    # Audio
    download_audio(target_dir, dataset_parts, force_download, url, mic)
    
    # Annotations
    text_name = 'annotations.gzip'
    text_path = target_dir / text_name
    if force_download or not text_path.is_file():
        text_url = f'{url}/AMICorpusAnnotations/ami_manual_annotations_v1.6.1_export.gzip'
        urllib.request.urlretrieve(text_url, filename=text_path)


class AmiSegmentAnnotation(NamedTuple):
    text: str
    speaker: str
    gender: str
    begin_time: Seconds
    end_time: Seconds


def parse_ami_annotations(gzip_file: Pathlike, mic: Optional[str]='ihm'
    ) -> Dict[str, List[AmiSegmentAnnotation]]:
    annotations = {}

    with GzipFile(gzip_file) as f:
        for line in f:
            line = line.decode()
            line = re.sub(r'\s+$', '', line)
            line = re.sub(r'\.\.+', '', line)
            if re.match(r'^Found', line) or re.match(r'^[Oo]bs', line):
                continue
            meet_id, x, speaker, channel, transcriber_start, transcriber_end, starttime, endtime, trans, puncts_times \
                = line.split('\t')
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

    return annotations

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
        mic: Optional[str] = 'ihm'
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    If word alignment archive is provided, also returns a Supervision manifest
    corresponding to alignments.
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

    annotations = parse_ami_annotations(data_dir / 'annotations.gzip', mic=mic)

    # Audio
    wav_dir = data_dir / 'wav_db'
    if mic == 'ihm':
        audio_paths = wav_dir.rglob('*Headset-?.wav')
        audio = prepare_audio_ihm(list(audio_paths), dataset_parts)
    elif mic == 'ihm-mix':
        audio_paths = wav_dir.rglob('*Mix-Headset.wav')
        audio = prepare_audio_other(list(audio_paths), dataset_parts)
    elif mic == 'sdm':
        audio_paths = wav_dir.rglob('*Array1-01.wav')
        audio = prepare_audio_other(list(audio_paths), dataset_parts)

    # Supervisions
    if mic == 'ihm':
        supervision = prepare_supervision_ihm(audio, annotations)
    else:
        supervision = prepare_supervision_other(audio, annotations)

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
