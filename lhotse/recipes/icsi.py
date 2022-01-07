"""
The data preparation recipe for the ICSI Meeting Corpus. It follows the Kaldi recipe
by Pawel Swietojanski: 
https://git.informatik.fh-nuernberg.de/poppto72658/kaldi/-/commit/d5815d3255bb62eacf2fba6314f194fe09966453

ICSI data comprises around 72 hours of natural, meeting-style overlapped English speech 
recorded at International Computer Science Institute (ICSI), Berkley. 
Speech is captured using the set of parallel microphones, including close-talk headsets,
and several distant independent microhones (i.e. mics that do not form any explicitly 
known geometry, see below for an example layout). Recordings are sampled at 16kHz.

See [1] for more details on ICSI, or [2,3] to access the data. 
The correponding paper describing the ICSI corpora is [4]

[1] http://www1.icsi.berkeley.edu/Speech/mr/
[2] LDC: LDC2004S02 for audio, and LDC2004T04 for transcripts (used in this recipe)
[3] http://groups.inf.ed.ac.uk/ami/icsi/ (free access, but for now only ihm data is available for download)
[4] A Janin, D Baron, J Edwards, D Ellis, D Gelbart, N Morgan, B Peskin,
    T Pfau, E Shriberg, A Stolcke, and C Wooters, The ICSI meeting corpus. 
    in Proc IEEE ICASSP, 2003, pp. 364-367


ICSI data did not come with any pre-defined splits for train/valid/eval sets as it was
mostly used as a training material for NIST RT evaluations. Some portions of the unrelased ICSI 
data (as a part of this corpora) can be found in, for example, NIST RT04 amd RT05 evaluation sets.

This recipe, however, to be self-contained factors out training (67.5 hours), development (2.2 hours 
and evaluation (2.8 hours) sets in a way to minimise the speaker-overlap between different partitions, 
and to avoid known issues with available recordings during evaluation. This recipe follows [5] where 
dev and eval sets are making use of {Bmr021, Bns00} and {Bmr013, Bmr018, Bro021} meetings, respectively.

[5] S Renals and P Swietojanski, Neural networks for distant speech recognition. 
    in Proc IEEE HSCMA 2014 pp. 172-176. DOI:10.1109/HSCMA.2014.6843274

Below description is (mostly) copied from ICSI documentation for convenience.
=================================================================================

Simple diagram of the seating arrangement in the ICSI meeting room.             
                                                                                
The ordering of seat numbers is as specified below, but their                   
alignment with microphones may not always be as precise as indicated            
here. Also, the seat number only indicates where the participant                
started the meeting. Since most of the microphones are wireless, they           
were able to move around.                                                       
                                                                                                                                                                
   Door                                                                         
                                                                                
                                                                                
          1         2            3           4                                  
     -----------------------------------------------------------------------    
     |                      |                       |                      |   S
     |                      |                       |                      |   c
     |                      |                       |                      |   r
    9|   D1        D2       |   D3  PDA     D4      |                      |   e
     |                      |                       |                      |   e
     |                      |                       |                      |   n
     |                      |                       |                      |    
     -----------------------------------------------------------------------    
          8         7            6           5                                  
                                                                                
                                                                                
                                                                                
D1, D2, D3, D4  - Desktop PZM microphones                                       
PDA - The mockup PDA with two cheap microphones                                 
                                                                                
The following are the TYPICAL channel assignments, although a handful           
of meetings (including Bmr003, Btr001, Btr002) differed in assignment.         

The mapping from the above, to the actual waveform channels in the corpora,
and (this recipe for a signle distant mic case) is:
                                                                                
D1 - chanE - (this recipe: sdm3)                                                                      
D2 - chanF - (this recipe: sdm4)                                                                     
D3 - chan6 - (this recipe: sdm1)                                                                     
D4 - chan7 - (this recipe: sdm2)                                                                     
PDA left - chanC                                                                
PDA right - chanD 

-----------
Note (Pawel): The mapping for headsets is being extracted from mrt files. 
In cases where IHM channels are missing for some speakers in some meetings, 
in this recipe we either back off to distant channel (typically D2, default)
or (optionally) skip this speaker's segments entirely from processing. 
This is not the case for eval set, where all the channels come with the 
expected recordings, and split is the same for all conditions (thus allowing 
for direct comparisons between IHM, SDM and MDM settings).
"""

import logging
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Union, Tuple
from lhotse.qa import fix_manifests

from tqdm.auto import tqdm

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import AudioSource, Recording, RecordingSet, read_sph
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, Seconds, urlretrieve_progress

# fmt:off
PARTITIONS = {
    'train': [
        "bdb001", "bed002", "bed003", "bed004", "bed005", "bed006", "bed008", "bed009", 
        "bed010", "bed011", "bed012", "bed013", "bed014", "bed015", "bed016", "bed017", 
        "bmr001", "bmr002", "bmr003", "bmr005", "bmr006", "bmr007", "bmr008", "bmr009", 
        "bmr010", "bmr011", "bmr012", "bmr014", "bmr015", "bmr016", "bmr019", "bmr020", 
        "bmr022", "bmr023", "bmr024", "bmr025", "bmr026", "bmr027", "bmr028", "bmr029", 
        "bmr030", "bmr031", "bns002", "bns003", "bro003", "bro004", "bro005", "bro007", 
        "bro008", "bro010", "bro011", "bro012", "bro013", "bro014", "bro015", "bro016", 
        "bro017", "bro018", "bro019", "bro022", "bro023", "bro024", "bro025", "bro026", 
        "bro027", "bro028", "bsr001", "btr001", "btr002", "buw001",
    ],
    'dev': ["bmr021", "bns001"],
    'test': ["bmr013", "bmr018", "bro021"]
}

MICS = ["ihm", "sdm", "mdm"]  # See AMI recipe for description of mic types
# fmt:on


class IcsiSegmentAnnotation(NamedTuple):
    text: str
    speaker: str
    channel: str
    gender: str
    start_time: Seconds
    end_time: Seconds


def parse_icsi_annotations(
    transcripts_dir: Pathlike, normalize_text: bool = True
) -> Tuple[Dict[str, List[SupervisionSegment]], Dict[str, Dict[str, int]]]:

    annotations = defaultdict(list)
    # In Lhotse, channels are integers, so we map channel ids to integers for each session
    channel_to_idx_map = defaultdict(dict)
    spk_to_channel_map = defaultdict(dict)

    # First we get global speaker ids and channels
    for meeting_file in tqdm(
        transcripts_dir.rglob("transcripts/*.mrt"), desc="Parsing ICSI mrt files"
    ):
        if meeting_file.stem == "preambles":
            continue
        with open(meeting_file) as f:
            meeting_id = meeting_file.stem.lower()
            root = ET.parse(f).getroot()  # <Meeting>
            for child in root:
                if child.tag == "Preamble":
                    for grandchild in child:
                        if grandchild.tag == "Channels":
                            channel_to_idx_map[meeting_id] = {
                                channel.attrib["Name"]: idx
                                for idx, channel in enumerate(grandchild)
                            }
                        elif grandchild.tag == "Participants":
                            for speaker in grandchild:
                                # some speakers may not have an associated channel in some meetings, so we
                                # assign them the SDM channel
                                spk_to_channel_map[meeting_id][
                                    speaker.attrib["Name"]
                                ] = (
                                    speaker.attrib["Channel"]
                                    if "Channel" in speaker.attrib
                                    else "chan6"
                                )
                elif child.tag == "Transcript":
                    for segment in child:
                        if len(list(segment)) == 0 and "Participant" in segment.attrib:
                            start_time = float(segment.attrib["StartTime"])
                            end_time = float(segment.attrib["EndTime"])
                            speaker = segment.attrib["Participant"]
                            channel = spk_to_channel_map[meeting_id][speaker]
                            text = (
                                segment.text.strip().upper()
                                if normalize_text
                                else segment.text.strip()
                            )
                            annotations[(meeting_id, speaker, channel)].append(
                                IcsiSegmentAnnotation(
                                    text,
                                    speaker,
                                    channel,
                                    speaker[0],
                                    start_time,
                                    end_time,
                                )
                            )
    return annotations, channel_to_idx_map


# IHM and MDM audio requires grouping multiple channels of AudioSource into
# one Recording.


def prepare_audio_grouped(
    audio_paths: List[Pathlike],
    channel_to_idx_map: Dict[str, Dict[str, int]] = None,
) -> RecordingSet:

    # Group together multiple channels from the same session.
    # We will use that to create a Recording with multiple sources (channels).
    from cytoolz import groupby

    channel_wavs = groupby(lambda p: p.parts[-2].lower(), audio_paths)

    if channel_to_idx_map is None:
        channel_to_idx_map = defaultdict(dict)
    recordings = []
    for session_name, channel_paths in tqdm(
        channel_wavs.items(), desc="Preparing audio"
    ):
        if session_name not in channel_to_idx_map:
            channel_to_idx_map[session_name] = {
                c: idx for idx, c in enumerate(["chanE", "chanF", "chan6", "chan7"])
            }
        audio_sf, samplerate = read_sph(channel_paths[0])

        recordings.append(
            Recording(
                id=session_name,
                sources=[
                    AudioSource(
                        type="file",
                        channels=[channel_to_idx_map[session_name][audio_path.stem]],
                        source=str(audio_path),
                    )
                    for audio_path in sorted(channel_paths)
                    if audio_path.stem in channel_to_idx_map[session_name]
                ],
                sampling_rate=samplerate,
                num_samples=audio_sf.shape[1],
                duration=audio_sf.shape[1] / samplerate,
            )
        )
    return RecordingSet.from_recordings(recordings)


# SDM setting does not require any grouping


def prepare_audio_single(
    audio_paths: List[Pathlike],
) -> RecordingSet:
    import soundfile as sf

    recordings = []
    for audio_path in tqdm(audio_paths, desc="Preparing audio"):
        session_name = audio_path.parts[-2].lower()
        audio_sf, samplerate = read_sph(audio_path)
        recordings.append(
            Recording(
                id=session_name,
                sources=[
                    AudioSource(
                        type="file",
                        channels=list(range(audio_sf.shape[0])),
                        source=str(audio_path),
                    )
                ],
                sampling_rate=samplerate,
                num_samples=audio_sf.shape[1],
                duration=audio_sf.shape[1] / samplerate,
            )
        )
    return RecordingSet.from_recordings(recordings)


# For IHM mic, each headphone will have its own annotations, while for other mics
# all sources have the same annotation


def prepare_supervision_ihm(
    audio: RecordingSet,
    annotations: Dict[str, List[IcsiSegmentAnnotation]],
    channel_to_idx_map: Dict[str, Dict[str, int]],
) -> SupervisionSet:
    # Create a mapping from a tuple of (session_id, channel) to the list of annotations.
    # This way we can map the supervisions to the right channels in a multi-channel recording.
    annotation_by_id_and_channel = {
        (key[0], channel_to_idx_map[key[0]][key[2]]): annotations[key]
        for key in annotations
    }

    segments = []
    for recording in tqdm(audio, desc="Preparing supervision"):
        # IHM can have multiple audio sources for each recording
        for source in recording.sources:
            # For each source, "channels" will always be a one-element list
            (channel,) = source.channels
            annotation = annotation_by_id_and_channel.get((recording.id, channel))

            if annotation is None:
                continue

            for seg_idx, seg_info in enumerate(annotation):
                duration = seg_info.end_time - seg_info.start_time
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
                            start=seg_info.start_time,
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
    audio: RecordingSet, annotations: Dict[str, List[IcsiSegmentAnnotation]]
) -> SupervisionSet:
    annotation_by_id = defaultdict(list)
    for key, value in annotations.items():
        annotation_by_id[key[0]].extend(value)

    segments = []
    for recording in tqdm(audio, desc="Preparing supervision"):
        annotation = annotation_by_id.get(recording.id)
        # In these mic settings, all sources will share supervision.
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
            duration = seg_info.end_time - seg_info.start_time
            if duration > 0:
                segments.append(
                    SupervisionSegment(
                        id=f"{recording.id}-{seg_idx}",
                        recording_id=recording.id,
                        start=seg_info.start_time,
                        duration=duration,
                        channel=source.channels[0],
                        language="English",
                        speaker=seg_info.speaker,
                        gender=seg_info.gender,
                        text=seg_info.text,
                    )
                )
    return SupervisionSet.from_segments(segments)


def prepare_icsi(
    audio_dir: Pathlike,
    transcripts_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    mic: Optional[str] = "ihm",
    normalize_text: bool = True,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param data_dir: Pathlike, the path of the audio dir (LDC2004S02).
    :param transcripts_dir: Pathlike, the path of the transcripts dir (LDC2004T04).
    :param output_dir: Pathlike, the path where to write the manifests.
    :param mic: str {'ihm','sdm','mdm'}, type of mic to use.
    :param normalize_text: bool, whether to normalize text to uppercase
    :return: a Dict whose key is ('train', 'dev', 'test'), and the values are dicts of manifests under keys
        'recordings' and 'supervisions'.
    """
    audio_dir = Path(audio_dir)
    transcripts_dir = Path(transcripts_dir)
    assert audio_dir.is_dir(), f"No such directory: {audio_dir}"
    assert transcripts_dir.is_dir(), f"No such directory: {transcripts_dir}"
    assert mic in MICS, f"Mic {mic} not supported"

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Parsing ICSI transcripts")
    annotations, channel_to_idx_map = parse_icsi_annotations(
        transcripts_dir, normalize_text
    )

    # Audio
    logging.info("Preparing recording manifests")

    if mic == "ihm":
        audio_paths = audio_dir.rglob("chan[1-9].sph")
        audio = prepare_audio_grouped(list(audio_paths), channel_to_idx_map)
    elif mic == "mdm":
        audio_paths = audio_dir.rglob("chan[EF67].sph")
        audio = prepare_audio_grouped(list(audio_paths))
    else:
        audio_paths = audio_dir.rglob("chan6.sph")
        audio = prepare_audio_single(list(audio_paths))

    # Supervisions
    logging.info("Preparing supervision manifests")
    supervision = (
        prepare_supervision_ihm(audio, annotations, channel_to_idx_map)
        if mic == "ihm"
        else prepare_supervision_other(audio, annotations)
    )

    manifests = defaultdict(dict)

    for part in ["train", "dev", "test"]:
        # Get recordings for current data split
        audio_part = audio.filter(lambda x: x.id in PARTITIONS[part])
        supervision_part = supervision.filter(
            lambda x: x.recording_id in PARTITIONS[part]
        )

        # Write to output directory if a path is provided
        if output_dir is not None:
            audio_part.to_file(output_dir / f"recordings_{part}.jsonl")
            supervision_part.to_file(output_dir / f"supervisions_{part}.jsonl")

        audio_part, supervision_part = fix_manifests(audio_part, supervision_part)
        validate_recordings_and_supervisions(audio_part, supervision_part)

        # Combine all manifests into one dictionary
        manifests[part] = {"recordings": audio_part, "supervisions": supervision_part}

    return dict(manifests)
