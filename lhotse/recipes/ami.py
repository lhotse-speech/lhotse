"""
The data preparation recipe for the AMI Meeting Corpus.

NOTE on data splits and references:

- The official AMI documentation (http://groups.inf.ed.ac.uk/ami/corpus/datasets.shtml) recommends
three different data partitions: scenario-only, full-corpus, and full-corpus-asr, based on the
task that the data is used for. We provide an argument `partition` which specifies which
partition is to be used.

- We use the latest version of the official annotations: ami_public_manual_1.6.2. This differs from
the Kaldi s5 and s5b recipes which use 1.6.1 (known to have alignment and annotation issues). We
get word-level annotations with time-marks and combine adjacent words into one segment if: (i) they
belong to the same speaker, and (ii) there is no pause between the words. (These supervisions can
later be modified to get larger super-segments based on the task)

NOTE on mic settings: AMI comes with 4 different microphone settings:

- ihm (individual headset microphone)
- sdm (single distant microphone)
- ihm-mix (mix-headset sum)
- mdm (multiple distant microphone)

These can be specified using the `mic` argument.
"""

import html
import itertools
import logging
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Union

from tqdm.auto import tqdm

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import AudioSource, Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, Seconds, urlretrieve_progress

# fmt: off
MEETINGS = {
    'EN2001': ['EN2001a', 'EN2001b', 'EN2001d', 'EN2001e'],
    'EN2002': ['EN2002a', 'EN2002b', 'EN2002c', 'EN2002d'],
    'EN2003': ['EN2003a'],
    'EN2004': ['EN2004a'],
    'EN2005': ['EN2005a'],
    'EN2006': ['EN2006a','EN2006b'],
    'EN2009': ['EN2009b','EN2009c','EN2009d'],
    'ES2002': ['ES2002a','ES2002b','ES2002c','ES2002d'],
    'ES2003': ['ES2003a','ES2003b','ES2003c','ES2003d'],
    'ES2004': ['ES2004a','ES2004b','ES2004c','ES2004d'],
    'ES2005': ['ES2005a','ES2005b','ES2005c','ES2005d'],
    'ES2006': ['ES2006a','ES2006b','ES2006c','ES2006d'],
    'ES2007': ['ES2007a','ES2007b','ES2007c','ES2007d'],
    'ES2008': ['ES2008a','ES2008b','ES2008c','ES2008d'],
    'ES2009': ['ES2009a','ES2009b','ES2009c','ES2009d'],
    'ES2010': ['ES2010a','ES2010b','ES2010c','ES2010d'],
    'ES2011': ['ES2011a','ES2011b','ES2011c','ES2011d'],
    'ES2012': ['ES2012a','ES2012b','ES2012c','ES2012d'],
    'ES2013': ['ES2013a','ES2013b','ES2013c','ES2013d'],
    'ES2014': ['ES2014a','ES2014b','ES2014c','ES2014d'],
    'ES2015': ['ES2015a','ES2015b','ES2015c','ES2015d'],
    'ES2016': ['ES2016a','ES2016b','ES2016c','ES2016d'],
    'IB4001': ['IB4001'],
    'IB4002': ['IB4002'],
    'IB4003': ['IB4003'],
    'IB4004': ['IB4004'],
    'IB4005': ['IB4005'],
    'IB4010': ['IB4010'],
    'IB4011': ['IB4011'],
    'IN1001': ['IN1001'],
    'IN1002': ['IN1002'],
    'IN1005': ['IN1005'],
    'IN1007': ['IN1007'],
    'IN1008': ['IN1008'],
    'IN1009': ['IN1009'],
    'IN1012': ['IN1012'],
    'IN1013': ['IN1013'],
    'IN1014': ['IN1014'],
    'IN1016': ['IN1016'],
    'IS1000': ['IS1000a','IS1000b','IS1000c','IS1000d'],
    'IS1001': ['IS1001a','IS1001b','IS1001c','IS1001d'],
    'IS1002': ['IS1002b','IS1002c','IS1002d'],
    'IS1003': ['IS1003a','IS1003b','IS1003c','IS1003d'],
    'IS1004': ['IS1004a','IS1004b','IS1004c','IS1004d'],
    'IS1005': ['IS1005a','IS1005b','IS1005c'],
    'IS1006': ['IS1006a','IS1006b','IS1006c','IS1006d'],
    'IS1007': ['IS1007a','IS1007b','IS1007c','IS1007d'],
    'IS1008': ['IS1008a','IS1008b','IS1008c','IS1008d'],
    'IS1009': ['IS1009a','IS1009b','IS1009c','IS1009d'],
    'TS3003': ['TS3003a','TS3003b','TS3003c','TS3003d'],
    'TS3004': ['TS3004a','TS3004b','TS3004c','TS3004d'],
    'TS3005': ['TS3005a','TS3005b','TS3005c','TS3005d'],
    'TS3006': ['TS3006a','TS3006b','TS3006c','TS3006d'],
    'TS3007': ['TS3007a','TS3007b','TS3007c','TS3007d'],
    'TS3008': ['TS3008a','TS3008b','TS3008c','TS3008d'],
    'TS3009': ['TS3009a','TS3009b','TS3009c','TS3009d'],
    'TS3010': ['TS3010a','TS3010b','TS3010c','TS3010d'],
    'TS3011': ['TS3011a','TS3011b','TS3011c','TS3011d'],
    'TS3012': ['TS3012a','TS3012b','TS3012c','TS3012d'],
}

PARTITIONS = {
    'scenario-only': {
        'train': [meeting for session in [
                'ES2002','ES2005','ES2006','ES2007','ES2008','ES2009','ES2010','ES2012','ES2013',
                'ES2015','ES2016','IS1000','IS1001','IS1002','IS1003','IS1004','IS1005','IS1006',
                'IS1007','TS3005','TS3008','TS3009','TS3010','TS3011','TS3012'
            ] for meeting in MEETINGS[session] if meeting not in ['IS1002a','IS1005d']],
        'dev': [meeting for session in [
                'ES2003','ES2011','IS1008','TS3004','TS3006'
            ] for meeting in MEETINGS[session]],
        'test': [meeting for session in [
                'ES2004','ES2014','IS1009','TS3003','TS3007'
            ] for meeting in MEETINGS[session]]
    },
    'full-corpus': {
        'train': [meeting for session in [
                'ES2002','ES2005','ES2006','ES2007','ES2008','ES2009','ES2010','ES2012','ES2013',
                'ES2015','ES2016','IS1000','IS1001','IS1002','IS1003','IS1004','IS1005','IS1006',
                'IS1007','TS3005','TS3008','TS3009','TS3010','TS3011','TS3012','EN2001','EN2003',
                'EN2004','EN2005','EN2006','EN2009','IN1001','IN1002','IN1005','IN1007','IN1008',
                'IN1009','IN1012','IN1013','IN1014','IN1016'
            ] for meeting in MEETINGS[session]],
        'dev': [meeting for session in [
                'ES2003','ES2011','IS1008','TS3004','TS3006','IB4001','IB4002','IB4003','IB4004',
                'IB4010','IB4011'
            ] for meeting in MEETINGS[session]],
        'test': [meeting for session in [
                'ES2004','ES2014','IS1009','TS3003','TS3007','EN2002'
            ] for meeting in MEETINGS[session]]
    },
    'full-corpus-asr': {
        'train': [meeting for session in [
                'ES2002','ES2005','ES2006','ES2007','ES2008','ES2009','ES2010','ES2012','ES2013',
                'ES2015','ES2016','IS1000','IS1001','IS1002','IS1003','IS1004','IS1005','IS1006',
                'IS1007','TS3005','TS3008','TS3009','TS3010','TS3011','TS3012','EN2001','EN2003',
                'EN2004','EN2005','EN2006','EN2009','IN1001','IN1002','IN1005','IN1007','IN1008',
                'IN1009','IN1012','IN1013','IN1014','IN1016','ES2014','TS3007','ES2003','TS3006'
            ] for meeting in MEETINGS[session]],
        'dev': [meeting for session in [
                'ES2011','IS1008','TS3004','IB4001','IB4002','IB4003','IB4004','IB4010','IB4011'
            ] for meeting in MEETINGS[session]],
        'test': [meeting for session in [
                'ES2004','IS1009','TS3003','EN2002'
            ] for meeting in MEETINGS[session]]
    }
}

MICS = ['ihm','ihm-mix','sdm','mdm']
MDM_ARRAYS = ['Array1','Array2']
MDM_CHANNELS = ['01','02','03','04','05','06','07','08']
# fmt: on


def download_audio(
    target_dir: Path,
    force_download: Optional[bool] = False,
    url: Optional[str] = "http://groups.inf.ed.ac.uk/ami",
    mic: Optional[str] = "ihm",
) -> None:
    # Audios
    for item in tqdm(
        itertools.chain.from_iterable(MEETINGS.values()),
        desc="Downloading AMI meetings",
    ):
        if mic == "ihm":
            headset_num = 5 if item in ("EN2001a", "EN2001d", "EN2001e") else 4
            for m in range(headset_num):
                wav_name = f"{item}.Headset-{m}.wav"
                wav_url = f"{url}/AMICorpusMirror/amicorpus/{item}/audio/{wav_name}"
                wav_dir = target_dir / "wav_db" / item / "audio"
                wav_dir.mkdir(parents=True, exist_ok=True)
                wav_path = wav_dir / wav_name
                if force_download or not wav_path.is_file():
                    urlretrieve_progress(
                        wav_url, filename=wav_path, desc=f"Downloading {wav_name}"
                    )
        elif mic == "ihm-mix":
            wav_name = f"{item}.Mix-Headset.wav"
            wav_url = f"{url}/AMICorpusMirror/amicorpus/{item}/audio/{wav_name}"
            wav_dir = target_dir / "wav_db" / item / "audio"
            wav_dir.mkdir(parents=True, exist_ok=True)
            wav_path = wav_dir / wav_name
            if force_download or not wav_path.is_file():
                urlretrieve_progress(
                    wav_url, filename=wav_path, desc=f"Downloading {wav_name}"
                )
        elif mic == "sdm":
            wav_name = f"{item}.Array1-01.wav"
            wav_url = f"{url}/AMICorpusMirror/amicorpus/{item}/audio/{wav_name}"
            wav_dir = target_dir / "wav_db" / item / "audio"
            wav_dir.mkdir(parents=True, exist_ok=True)
            wav_path = wav_dir / wav_name
            if force_download or not wav_path.is_file():
                urlretrieve_progress(
                    wav_url, filename=wav_path, desc=f"Downloading {wav_name}"
                )
        elif mic == "mdm":
            for array in MDM_ARRAYS:
                for channel in MDM_CHANNELS:
                    wav_name = f"{item}.{array}-{channel}.wav"
                    wav_url = f"{url}/AMICorpusMirror/amicorpus/{item}/audio/{wav_name}"
                    wav_dir = target_dir / "wav_db" / item / "audio"
                    wav_dir.mkdir(parents=True, exist_ok=True)
                    wav_path = wav_dir / wav_name
                    if force_download or not wav_path.is_file():
                        urlretrieve_progress(
                            wav_url, filename=wav_path, desc=f"Downloading {wav_name}"
                        )


def download_ami(
    target_dir: Pathlike = ".",
    annotations_dir: Optional[Pathlike] = None,
    force_download: Optional[bool] = False,
    url: Optional[str] = "http://groups.inf.ed.ac.uk/ami",
    mic: Optional[str] = "ihm",
) -> None:
    """
    Download AMI audio and annotations for provided microphone setting.
    :param target_dir: Pathlike, the path to store the data.
    :param annotations_dir: Pathlike (default = None), path to save annotations zip file
    :param force_download: bool (default = False), if True, download even if file is present.
    :param url: str (default = 'http://groups.inf.ed.ac.uk/ami'), AMI download URL.
    :param mic: str {'ihm','ihm-mix','sdm','mdm'}, type of mic setting.
    """
    target_dir = Path(target_dir)

    annotations_dir = target_dir if not annotations_dir else annotations_dir

    # Audio
    download_audio(target_dir, force_download, url, mic)

    # Annotations
    logging.info("Downloading AMI annotations")
    annotations_name = "annotations.zip"
    annotations_path = annotations_dir / annotations_name
    if annotations_path.exists():
        logging.info(f'Skip downloading annotations as they exist in: {annotations_path}')
        return
    annotations_url = f"{url}/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
    if force_download or not annotations_path.is_file():
        urllib.request.urlretrieve(annotations_url, filename=annotations_path)


class AmiSegmentAnnotation(NamedTuple):
    text: str
    speaker: str
    gender: str
    begin_time: Seconds
    end_time: Seconds


def parse_ami_annotations(
    annotations_zip: Pathlike, max_pause: float
) -> Dict[str, List[SupervisionSegment]]:
    annotations = defaultdict(dict)
    with zipfile.ZipFile(annotations_zip, "r") as archive:
        # First we get global speaker ids and channels
        global_spk_id = {}
        channel_id = {}
        with archive.open("corpusResources/meetings.xml") as f:
            tree = ET.parse(f)
            for meeting in tree.getroot():
                meet_id = meeting.attrib["observation"]
                for speaker in meeting:
                    local_id = (meet_id, speaker.attrib["nxt_agent"])
                    global_spk_id[local_id] = speaker.attrib["global_name"]
                    channel_id[local_id] = int(speaker.attrib["channel"])

        # Now iterate over all alignments
        for file in archive.namelist():
            if file.startswith("words/") and file[-1] != "/":
                meet_id, x, _, _ = file.split("/")[1].split(".")
                if (meet_id, x) not in global_spk_id:
                    logging.warning(
                        f"No speaker {meet_id}.{x} found! Skipping annotation."
                    )
                    continue
                spk = global_spk_id[(meet_id, x)]
                channel = channel_id[(meet_id, x)]
                tree = ET.parse(archive.open(file))
                key = (meet_id, spk, channel)
                if key not in annotations:
                    annotations[key] = []
                for child in tree.getroot():
                    # If the alignment does not contain start or end time info,
                    # ignore them. Also, only consider words in the alignment XML files.
                    if (
                        "starttime" not in child.attrib
                        or "endtime" not in child.attrib
                        or child.tag != "w"
                    ):
                        continue
                    text = child.text if child.tag == "w" else child.attrib["type"]
                    # to convert HTML escape sequences
                    text = html.unescape(text)
                    annotations[key].append(
                        AmiSegmentAnnotation(
                            text=text,
                            speaker=spk,
                            gender=spk[0],
                            begin_time=float(child.attrib["starttime"]),
                            end_time=float(child.attrib["endtime"]),
                        )
                    )

        # Post-process segments and combine neighboring segments from the same speaker
        for key in annotations:
            new_segs = []
            cur_seg = list(annotations[key])[0]
            for seg in list(annotations[key])[1:]:
                if seg.begin_time - cur_seg.end_time <= max_pause:
                    cur_seg._replace(
                        text=f"{cur_seg.text} {seg.text}", end_time=seg.end_time
                    )
                else:
                    new_segs.append(cur_seg)
                    cur_seg = seg
            new_segs.append(cur_seg)
            annotations[key] = new_segs
    return annotations


# IHM and MDM audio requires grouping multiple channels of AudioSource into
# one Recording.


def prepare_audio_grouped(
    audio_paths: List[Pathlike],
) -> RecordingSet:
    import soundfile as sf

    # Group together multiple channels from the same session.
    # We will use that to create a Recording with multiple sources (channels).
    from cytoolz import groupby

    channel_wavs = groupby(lambda p: p.parts[-3], audio_paths)

    recordings = []
    for session_name, channel_paths in channel_wavs.items():
        audio_sf = sf.SoundFile(str(channel_paths[0]))

        recordings.append(
            Recording(
                id=session_name,
                sources=[
                    AudioSource(type="file", channels=[idx], source=str(audio_path))
                    for idx, audio_path in enumerate(sorted(channel_paths))
                ],
                sampling_rate=audio_sf.samplerate,
                num_samples=audio_sf.frames,
                duration=audio_sf.frames / audio_sf.samplerate,
            )
        )
    return RecordingSet.from_recordings(recordings)


# SDM and IHM-Mix settings do not require any grouping


def prepare_audio_single(
    audio_paths: List[Pathlike],
) -> RecordingSet:
    import soundfile as sf

    recording_manifest = defaultdict(dict)

    recordings = []
    for audio_path in audio_paths:
        session_name = audio_path.parts[-3]
        audio_sf = sf.SoundFile(str(audio_path))
        recordings.append(
            Recording(
                id=session_name,
                sources=[
                    AudioSource(
                        type="file",
                        channels=list(range(audio_sf.channels)),
                        source=str(audio_path),
                    )
                ],
                sampling_rate=audio_sf.samplerate,
                num_samples=audio_sf.frames,
                duration=audio_sf.frames / audio_sf.samplerate,
            )
        )
    return RecordingSet.from_recordings(recordings)


# For IHM mic, each headphone will have its own annotations, while for other mics
# all sources have the same annotation


def prepare_supervision_ihm(
    audio: RecordingSet, annotations: Dict[str, List[AmiSegmentAnnotation]]
) -> SupervisionSet:
    # Create a mapping from a tuple of (session_id, channel) to the list of annotations.
    # This way we can map the supervisions to the right channels in a multi-channel recording.
    annotation_by_id_and_channel = {
        (key[0], key[2]): annotations[key] for key in annotations
    }

    segments = []
    for recording in audio:
        # AMI IHM can have multiple audio sources for each recording
        for source in recording.sources:
            # For each source, "channels" will always be a one-element list
            (channel,) = source.channels
            annotation = annotation_by_id_and_channel.get((recording.id, channel))
            if annotation is None:
                logging.warning(
                    f"No annotation found for recording {recording.id} "
                    f"(file {source.source})"
                )
                continue

            for seg_idx, seg_info in enumerate(annotation):
                duration = seg_info.end_time - seg_info.begin_time
                # Some annotations in IHM setting exceed audio duration, so we
                # ignore such segments
                if seg_info.end_time > recording.duration:
                    logging.warning(
                        f"Segment {recording.id}-{channel}-{seg_idx} exceeds "
                        f"recording duration. Not adding to supervisions."
                    )
                    continue
                if duration > 0:
                    segments.append(
                        SupervisionSegment(
                            id=f"{recording.id}-{channel}-{seg_idx}",
                            recording_id=recording.id,
                            start=seg_info.begin_time,
                            duration=duration,
                            channel=channel,
                            language="English",
                            speaker=seg_info.speaker,
                            gender=seg_info.gender,
                            text=seg_info.text,
                        )
                    )

    return SupervisionSet.from_segments(segments)


def prepare_supervision_other(
    audio: RecordingSet, annotations: Dict[str, List[AmiSegmentAnnotation]]
) -> SupervisionSet:
    annotation_by_id = {(key[0]): annot for key, annot in annotations.items()}

    segments = []
    for recording in audio:
        annotation = annotation_by_id.get(recording.id)
        # In these mic settings, all sources (1 for ihm-mix and sdm and 16 for mdm)
        # will share supervision.
        source = recording.sources[0]
        if annotation is None:
            logging.warning(f"No annotation found for recording {recording.id}")
            continue

        if len(source.channels) > 1:
            logging.warning(
                f"More than 1 channels in recording {recording.id}. "
                f"Creating supervision for channel 0 only."
            )

        for seg_idx, seg_info in enumerate(annotation):
            duration = seg_info.end_time - seg_info.begin_time
            if duration > 0:
                segments.append(
                    SupervisionSegment(
                        id=f"{recording.id}-{seg_idx}",
                        recording_id=recording.id,
                        start=seg_info.begin_time,
                        duration=duration,
                        channel=0,
                        language="English",
                        speaker=seg_info.speaker,
                        gender=seg_info.gender,
                        text=seg_info.text,
                    )
                )
    return SupervisionSet.from_segments(segments)


def prepare_ami(
    data_dir: Pathlike,
    annotations_dir: Optional[Pathlike] = None,
    output_dir: Optional[Pathlike] = None,
    mic: Optional[str] = "ihm",
    partition: Optional[str] = "full-corpus",
    max_pause: Optional[float] = 0.0,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param data_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param mic: str {'ihm','ihm-mix','sdm','mdm'}, type of mic to use.
    :param partition: str {'full-corpus','full-corpus-asr','scenario-only'}, AMI official data split
    :param max_pause: float (default = 0.0), max pause allowed between word segments to combine segments
    :return: a Dict whose key is ('train', 'dev', 'eval'), and the values are dicts of manifests under keys
        'recordings' and 'supervisions'.

    The `partition` and `max_pause` must be chosen depending on the task. For example:
    - Speaker diarization: set `partition="full-corpus"` and `max_pause=0`
    - ASR: set `partition="full-corpus-asr"` and `max_pause=0.3` (or some value in the range 0.2-0.5)
    """
    data_dir = Path(data_dir)
    assert data_dir.is_dir(), f"No such directory: {data_dir}"
    assert mic in MICS, f"Mic {mic} not supported"
    assert partition in PARTITIONS, f"Partition {partition} not supported"

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Parsing AMI annotations")
    annotations_dir = data_dir if not annotations_dir else annotations_dir
    annotations = parse_ami_annotations(
        Path(annotations_dir) / "annotations.zip", max_pause=max_pause
    )

    # Audio
    logging.info("Preparing recording manifests")
    wav_dir = data_dir

    if mic in ["ihm", "mdm"]:
        audio_paths = (
            wav_dir.rglob("*Headset-?.wav")
            if mic == "ihm"
            else wav_dir.rglob("*Array?-0?.wav")
        )
        audio = prepare_audio_grouped(list(audio_paths))
    elif mic in ["ihm-mix", "sdm"]:
        audio_paths = (
            wav_dir.rglob("*Mix-Headset.wav")
            if mic == "ihm-mix"
            else wav_dir.rglob("*Array1-01.wav")
        )
        audio = prepare_audio_single(list(audio_paths))

    # Supervisions
    logging.info("Preparing supervision manifests")
    supervision = (
        prepare_supervision_ihm(audio, annotations)
        if mic == "ihm"
        else prepare_supervision_other(audio, annotations)
    )

    manifests = defaultdict(dict)

    dataset_parts = PARTITIONS[partition]
    for part in ["train", "dev", "test"]:
        # Get recordings for current data split
        audio_part = audio.filter(lambda x: x.id in dataset_parts[part])
        supervision_part = supervision.filter(
            lambda x: x.recording_id in dataset_parts[part]
        )

        # Write to output directory if a path is provided
        if output_dir is not None:
            audio_part.to_json(output_dir / f"recordings_{part}.json")
            supervision_part.to_json(output_dir / f"supervisions_{part}.json")

        validate_recordings_and_supervisions(audio_part, supervision_part)

        # Combine all manifests into one dictionary
        manifests[part] = {"recordings": audio_part, "supervisions": supervision_part}

    return dict(manifests)
