"""
The data preparation recipe for the ICSI Meeting Corpus. It follows the Kaldi recipe
by Pawel Swietojanski: 
https://git.informatik.fh-nuernberg.de/poppto72658/kaldi/-/commit/d5815d3255bb62eacf2fba6314f194fe09966453

ICSI data comprises around 72 hours of natural, meeting-style overlapped English speech 
recorded at International Computer Science Institute (ICSI), Berkley. 
Speech is captured using the set of parallel microphones, including close-talk headsets,
and several distant independent microhones (i.e. mics that do not form any explicitly 
known geometry, see below for an example layout). Recordings are sampled at 16kHz.
 
The correponding paper describing the ICSI corpora is [1]

[1] A Janin, D Baron, J Edwards, D Ellis, D Gelbart, N Morgan, B Peskin,
    T Pfau, E Shriberg, A Stolcke, and C Wooters, The ICSI meeting corpus. 
    in Proc IEEE ICASSP, 2003, pp. 364-367


ICSI data did not come with any pre-defined splits for train/valid/eval sets as it was
mostly used as a training material for NIST RT evaluations. Some portions of the unrelased ICSI 
data (as a part of this corpora) can be found in, for example, NIST RT04 amd RT05 evaluation sets.

This recipe, however, to be self-contained factors out training (67.5 hours), development (2.2 hours 
and evaluation (2.8 hours) sets in a way to minimise the speaker-overlap between different partitions, 
and to avoid known issues with available recordings during evaluation. This recipe follows [2] where 
dev and eval sets are making use of {Bmr021, Bns001} and {Bmr013, Bmr018, Bro021} meetings, respectively.

[2] S Renals and P Swietojanski, Neural networks for distant speech recognition. 
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

NOTE on data: The ICSI data is freely available from the website (see `download` below)
and also as LDC corpora. The annotations that we download below are same as 
LDC2004T04, but there are some differences in the audio data, specifically in the
session names. Some sessions (Bns...) are named (bns...) in the LDC corpus, and the
Mix-Headset wav files are not available from the LDC corpus. So we recommend downloading
the public version even if you have an LDC subscription. The public data also includes
annotations of roles, dialog, summary etc. but we have not included them in this recipe.
"""

import logging
import itertools
import zipfile
import urllib
import ssl
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
from lhotse.recipes.ami import normalize_text

# fmt:off
PARTITIONS = {
    'train': [
        "Bdb001", "Bed002", "Bed003", "Bed004", "Bed005", "Bed006", "Bed008", "Bed009", 
        "Bed010", "Bed011", "Bed012", "Bed013", "Bed014", "Bed015", "Bed016", "Bed017", 
        "Bmr001", "Bmr002", "Bmr003", "Bmr005", "Bmr006", "Bmr007", "Bmr008", "Bmr009", 
        "Bmr010", "Bmr011", "Bmr012", "Bmr014", "Bmr015", "Bmr016", "Bmr019", "Bmr020", 
        "Bmr022", "Bmr023", "Bmr024", "Bmr025", "Bmr026", "Bmr027", "Bmr028", "Bmr029", 
        "Bmr030", "Bmr031", "Bns002", "Bns003", "Bro003", "Bro004", "Bro005", "Bro007", 
        "Bro008", "Bro010", "Bro011", "Bro012", "Bro013", "Bro014", "Bro015", "Bro016", 
        "Bro017", "Bro018", "Bro019", "Bro022", "Bro023", "Bro024", "Bro025", "Bro026", 
        "Bro027", "Bro028", "Bsr001", "Btr001", "Btr002", "Buw001",
    ],
    'dev': ["Bmr021", "Bns001"],
    'test': ["Bmr013", "Bmr018", "Bro021"]
}

MIC_TO_CHANNELS = {
    "ihm": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B"], 
    "sdm": ["6"],
    "mdm": ["E", "F", "6", "7"],
    "ihm-mix": [],
}
# fmt:on


class IcsiSegmentAnnotation(NamedTuple):
    text: str
    speaker: str
    channel: str
    gender: str
    start_time: Seconds
    end_time: Seconds


def download_audio(
    target_dir: Path,
    force_download: Optional[bool] = False,
    url: Optional[str] = "http://https://groups.inf.ed.ac.uk/ami",
    mic: Optional[str] = "ihm",
) -> None:
    # Audios
    for item in tqdm(
        itertools.chain.from_iterable(PARTITIONS.values()),
        desc="Downloading ICSI meetings",
    ):
        if mic in ["ihm", "sdm", "mdm"]:
            for channel in MIC_TO_CHANNELS[mic]:
                wav_url = f"{url}/ICSIsignals/SPH/{item}/chan{channel}.sph"
                wav_dir = target_dir / item
                wav_dir.mkdir(parents=True, exist_ok=True)
                wav_path = wav_dir / f"chan{channel}.sph"
                if force_download or not wav_path.is_file():
                    try:
                        urlretrieve_progress(
                            wav_url,
                            filename=wav_path,
                            desc=f"Downloading {item} chan{channel}.sph",
                        )
                    except urllib.error.HTTPError as e:
                        pass
        else:
            wav_url = f"{url}/ICSIsignals/NXT/{item}.interaction.wav"
            wav_dir = target_dir / item
            wav_dir.mkdir(parents=True, exist_ok=True)
            wav_path = wav_dir / f"Mix-Headset.wav"
            if force_download or not wav_path.is_file():
                urlretrieve_progress(
                    wav_url,
                    filename=wav_path,
                    desc=f"Downloading {item} Mix-Headset.wav",
                )


def download_icsi(
    target_dir: Pathlike = ".",
    audio_dir: Optional[Pathlike] = None,
    transcripts_dir: Optional[Pathlike] = None,
    force_download: Optional[bool] = False,
    url: Optional[str] = "http://groups.inf.ed.ac.uk/ami",
    mic: Optional[str] = "ihm",
) -> Path:
    """
    Download ICSI audio and annotations for provided microphone setting.
    :param target_dir: Pathlike, the path in which audio and transcripts dir are created by default.
    :param audio_dir: Pathlike (default = '<target_dir>/audio'), the path to store the audio data.
    :param transcripts_dir: Pathlike (default = '<target_dir>/transcripts'), path to store the transcripts data
    :param force_download: bool (default = False), if True, download even if file is present.
    :param url: str (default = 'http://groups.inf.ed.ac.uk/ami'), download URL.
    :param mic: str {'ihm','ihm-mix','sdm','mdm'}, type of mic setting.
    :return: the path to downloaded and extracted directory with data.
    """
    target_dir = Path(target_dir)
    audio_dir = Path(audio_dir) if audio_dir else target_dir / "speech"
    transcripts_dir = (
        Path(transcripts_dir) if transcripts_dir else target_dir / "transcripts"
    )

    # Audio
    download_audio(audio_dir, force_download, url, mic)

    # Annotations
    logging.info("Downloading AMI annotations")

    if (transcripts_dir).exists() and not force_download:
        logging.info(
            f"Skip downloading transcripts as they exist in: {transcripts_dir}"
        )
        return target_dir
    annotations_url = f"{url}/ICSICorpusAnnotations/ICSI_original_transcripts.zip"

    # The following is analogous to `wget --no-check-certificate``
    context = ssl._create_unverified_context()
    urllib.request.urlretrieve(
        annotations_url, filename=target_dir / "ICSI_original_transcripts.zip"
    )

    # Unzip annotations zip file
    with zipfile.ZipFile(target_dir / "ICSI_original_transcripts.zip") as z:
        # Unzips transcripts to <target_dir>/'transcripts'
        # zip file also contains some documentation which will be unzipped to <target_dir>
        z.extractall(target_dir)
        # If custom dir is passed, rename 'transcripts' dir accordingly
        if transcripts_dir:
            Path(target_dir / "transcripts").rename(transcripts_dir)

    return target_dir


def parse_icsi_annotations(
    transcripts_dir: Pathlike, normalize: str = "upper"
) -> Tuple[Dict[str, List[SupervisionSegment]], Dict[str, Dict[str, int]]]:

    annotations = defaultdict(list)
    # In Lhotse, channels are integers, so we map channel ids to integers for each session
    channel_to_idx_map = defaultdict(dict)
    spk_to_channel_map = defaultdict(dict)

    # First we get global speaker ids and channels
    for meeting_file in tqdm(
        transcripts_dir.rglob("./*.mrt"), desc="Parsing ICSI mrt files"
    ):
        if meeting_file.stem == "preambles":
            continue
        with open(meeting_file) as f:
            meeting_id = meeting_file.stem
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
                            text = normalize_text(
                                segment.text.strip(), normalize=normalize
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

    channel_wavs = groupby(lambda p: p.parts[-2], audio_paths)

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


# SDM and IHM-Mix settings do not require any grouping


def prepare_audio_single(
    audio_paths: List[Pathlike],
) -> RecordingSet:
    import soundfile as sf

    recordings = []
    for audio_path in tqdm(audio_paths, desc="Preparing audio"):
        session_name = audio_path.parts[-2]
        if audio_path.suffix == ".wav":
            audio_sf = sf.SoundFile(str(audio_path))
            num_frames = audio_sf.frames
            num_channels = audio_sf.channels
            samplerate = audio_sf.samplerate
        else:
            audio_sf, samplerate = read_sph(audio_path)
            num_channels, num_frames = audio_sf.shape
        recordings.append(
            Recording(
                id=session_name,
                sources=[
                    AudioSource(
                        type="file",
                        channels=list(range(num_channels)),
                        source=str(audio_path),
                    )
                ],
                sampling_rate=samplerate,
                num_samples=num_frames,
                duration=num_frames / samplerate,
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
                f"Skipping this recording."
            )
            continue

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
    normalize_text: str = "kaldi",
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param audio_dir: Pathlike, the path which holds the audio data
    :param transcripts_dir: Pathlike, the path which holds the transcripts data
    :param output_dir: Pathlike, the path where to write the manifests - `None` means manifests aren't stored on disk.
    :param mic: str {'ihm','ihm-mix','sdm','mdm'}, type of mic to use.
    :param normalize_text: str {'none', 'upper', 'kaldi'} normalization of text
    :return: a Dict whose key is ('train', 'dev', 'test'), and the values are dicts of manifests under keys
        'recordings' and 'supervisions'.
    """
    audio_dir = Path(audio_dir)
    transcripts_dir = Path(transcripts_dir)

    assert audio_dir.is_dir(), f"No such directory: {audio_dir}"
    assert transcripts_dir.is_dir(), f"No such directory: {transcripts_dir}"
    assert mic in MIC_TO_CHANNELS.keys(), f"Mic {mic} not supported"

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Parsing ICSI transcripts")
    annotations, channel_to_idx_map = parse_icsi_annotations(
        transcripts_dir, normalize=normalize_text
    )

    # Audio
    logging.info("Preparing recording manifests")

    channels = "".join(MIC_TO_CHANNELS[mic])
    if mic == "ihm" or mic == "mdm":
        audio_paths = audio_dir.rglob(f"chan[{channels}].sph")
        audio = prepare_audio_grouped(
            list(audio_paths), channel_to_idx_map if mic == "ihm" else None
        )
    elif mic == "sdm" or mic == "ihm-mix":
        audio_paths = (
            audio_dir.rglob(f"chan[{channels}].sph")
            if len(channels)
            else audio_dir.rglob("*.wav")
        )
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
