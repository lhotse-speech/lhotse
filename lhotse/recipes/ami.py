import logging
import os
import re
import urllib.request
from gzip import GzipFile
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Union

# Workaround for SoundFile (torchaudio dep) raising exception when a native library, libsndfile1, is not installed.
# Read-the-docs does not allow to modify the Docker containers used to build documentation...
if not os.environ.get('READTHEDOCS', False):
    import torchaudio

from lhotse.audio import AudioSource, Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, Seconds

# TODO: support other microphone settings like "sdm1" or "mdm8"
mic = 'ihm'

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
        url: Optional[str] = 'http://groups.inf.ed.ac.uk/ami'
) -> None:
    assert mic == 'ihm'
    target_dir = Path(target_dir)

    # Audios
    for part in dataset_parts:
        for item in dataset_parts[part]:
            headset_num = 5 if item in ('EN2001a', 'EN2001d', 'EN2001e') else 4
            for m in range(headset_num):
                wav_name = f'{item}.Headset-{m}.wav'
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


class AmiSegmentAnnotation(NamedTuple):
    text: str
    begin_time: Seconds
    end_time: Seconds


def parse_ami_annotations(gzip_file: Pathlike) -> Dict[str, List[AmiSegmentAnnotation]]:
    anotations = {}
    with GzipFile(gzip_file) as f:
        for line in f:
            line = re.sub(r'\s+$', '', line.decode())
            if re.match(r'^Found', line) or re.match(r'^[Oo]bs', line):
                continue
            meet_id, _, _, channel, _, _, aut_btime, aut_etime, trans, puncts_times = line.split('\t')

            # Split transcript by punctuations
            trans = re.split(r' *[.,?!:] *', re.sub(r'[.,?!:\s]+$', '', trans.upper()))

            # Process time points
            if puncts_times == '-':
                puncts_times = aut_etime
            try:
                aut_btime = float(aut_btime)
                aut_etime = float(aut_etime)
                puncts_times = [float(t) for t in puncts_times.split()]
                seg_num = len(trans)
                assert seg_num > 0
                if len(puncts_times) == seg_num - 1:
                    assert(puncts_times[-1] <= aut_etime)
                    puncts_times.append(aut_etime)
                if len(puncts_times) == seg_num + 1:
                    puncts_times.pop()
                    assert(puncts_times[-1] <= aut_etime)
                assert len(puncts_times) == seg_num and aut_btime <= puncts_times[0]
            except (ValueError, AssertionError):
                continue

            # Make the begin/end points list
            seg_btimes = [aut_btime] + puncts_times
            seg_btimes.pop()
            seg_times = [AmiSegmentAnnotation(t, b, e) for t, b, e in zip(trans, seg_btimes, puncts_times)]

            wav_name = f'{meet_id}.Headset-{int(channel)}.wav'
            if wav_name not in anotations:
                anotations[wav_name] = []
            anotations[wav_name].append(seg_times)

    return anotations


def prepare_ami(
        data_dir: Pathlike,
        output_dir: Pathlike,
        write_yaml: Optional[bool] = False,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    anotation_lists = parse_ami_annotations(data_dir / 'annotations.gzip')
    wav_dir = data_dir / 'wav_db'

    # Audio
    recordings = []
    for audio_path in wav_dir.rglob('*.wav'):
        audio_idx = audio_path.name
        if audio_idx not in anotation_lists:
            logging.warning(f'No annotation found for {audio_idx}')
            continue
        audio_info = torchaudio.info(str(audio_path))[0]

        recordings.append(Recording(
            id=audio_idx,
            sources=[
                AudioSource(
                    type='file',
                    channel_ids=[0],
                    source=str(audio_path)
                )
            ],
            sampling_rate=int(audio_info.rate),
            num_samples=audio_info.length,
            duration_seconds=int(audio_info.length / audio_info.rate),
        ))
    audio = RecordingSet.from_recordings(recordings)

    # Supervisions
    segments_by_pause = []
    for idx in audio.recordings:
        anotation = anotation_lists[idx]
        for seg_idx, seg_info in enumerate(anotation):
            for subseg_idx, subseg_info in enumerate(seg_info):
                duration = subseg_info.end_time - subseg_info.begin_time
                if duration > 0:
                    segments_by_pause.append(SupervisionSegment(
                        id=f'{audio_idx}-{seg_idx}-{subseg_idx}',
                        recording_id=idx,
                        start=subseg_info.begin_time,
                        duration=duration,
                        channel_id=0,
                        language='English',
                        speaker=re.sub(r'-.*', r'', idx),
                        text=subseg_info.text
                    ))
    supervision = SupervisionSet.from_segments(segments_by_pause)

    manifests = {
        'audio': audio,
        'supervisions': supervision
    }

    return manifests
