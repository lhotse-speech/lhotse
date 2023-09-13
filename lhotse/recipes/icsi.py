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

import itertools
import logging
import urllib
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import soundfile as sf
from tqdm.auto import tqdm

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import AudioSource, Recording, RecordingSet
from lhotse.audio.backend import read_sph
from lhotse.qa import fix_manifests
from lhotse.recipes.utils import normalize_text_ami
from lhotse.supervision import AlignmentItem, SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, Seconds, add_durations, resumable_download

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
                try:
                    resumable_download(
                        wav_url, filename=wav_path, force_download=force_download
                    )
                except urllib.error.HTTPError as e:
                    logging.warning(f"Skipping failed download from {wav_url}")
        else:
            wav_url = f"{url}/ICSIsignals/NXT/{item}.interaction.wav"
            wav_dir = target_dir / item
            wav_dir.mkdir(parents=True, exist_ok=True)
            wav_path = wav_dir / f"Mix-Headset.wav"
            resumable_download(
                wav_url, filename=wav_path, force_download=force_download
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
    logging.info("Downloading ICSI annotations")

    if transcripts_dir.exists() and not force_download:
        logging.info(
            f"Skip downloading transcripts as they exist in: {transcripts_dir}"
        )
        return target_dir

    # We need the MRT transcripts for the speaker-to-channel mapping. The NXT transcripts
    # are used for the actual annotations (since they contain word alignments)
    annotations_url_mrt = f"{url}/ICSICorpusAnnotations/ICSI_original_transcripts.zip"
    annotations_url_nxt = f"{url}/ICSICorpusAnnotations/ICSI_core_NXT.zip"
    resumable_download(
        annotations_url_mrt,
        filename=target_dir / "ICSI_original_transcripts.zip",
        force_download=force_download,
    )
    resumable_download(
        annotations_url_nxt,
        filename=target_dir / "ICSI_core_NXT.zip",
        force_download=force_download,
    )

    with zipfile.ZipFile(target_dir / "ICSI_core_NXT.zip") as z:
        # Unzips transcripts to <target_dir>/'transcripts'
        # zip file also contains some documentation which will be unzipped to <target_dir>
        z.extractall(target_dir)
        # If custom dir is passed, rename 'transcripts' dir accordingly
        if transcripts_dir:
            Path(target_dir / "transcripts").rename(transcripts_dir)

    # From the MRT transcripts, we only need the transcripts/preambles.mrt file
    with zipfile.ZipFile(target_dir / "ICSI_original_transcripts.zip") as z:
        z.extract("transcripts/preambles.mrt", transcripts_dir)

    return target_dir


class IcsiSegmentAnnotation(NamedTuple):
    text: str
    speaker: str
    gender: str
    start_time: Seconds
    end_time: Seconds
    words: List[AlignmentItem]


def parse_icsi_annotations(
    transcripts_dir: Pathlike, normalize: str = "upper"
) -> Tuple[
    Dict[Tuple[str, str, str], List[SupervisionSegment]], Dict[str, Dict[str, int]]
]:
    annotations = defaultdict(list)
    # In Lhotse, channels are integers, so we map channel ids to integers for each session
    channel_to_idx_map = defaultdict(dict)
    spk_to_channel_map = defaultdict(dict)

    # First we get global speaker ids and channels
    with open(transcripts_dir / "preambles.mrt") as f:
        root = ET.parse(f).getroot()  # <Meetings>
        for child in root:
            if child.tag == "Meeting":
                meeting_id = child.attrib["Session"]
                for grandchild in child:
                    if grandchild.tag == "Preamble":
                        for greatgrandchild in grandchild:
                            if greatgrandchild.tag == "Channels":
                                channel_to_idx_map[meeting_id] = {
                                    channel.attrib["Name"]: idx
                                    for idx, channel in enumerate(greatgrandchild)
                                }
                            elif greatgrandchild.tag == "Participants":
                                for speaker in greatgrandchild:
                                    # some speakers may not have an associated channel in some meetings, so we
                                    # assign them the SDM channel
                                    spk_to_channel_map[meeting_id][
                                        speaker.attrib["Name"]
                                    ] = (
                                        speaker.attrib["Channel"]
                                        if "Channel" in speaker.attrib
                                        else "chan6"
                                    )

    # Get the speaker segment times from the segments file
    segments = {}
    for file in (transcripts_dir / "Segments").glob("*.xml"):
        meet_id, local_id, _ = file.stem.split(".")
        spk_segments = []
        spk_id = None
        with open(file) as f:
            tree = ET.parse(f)
            for seg in tree.getroot():
                if seg.tag != "segment":
                    continue
                if spk_id is None and "participant" in seg.attrib:
                    spk_id = seg.attrib["participant"]
                start_time = float(seg.attrib["starttime"])
                end_time = float(seg.attrib["endtime"])
                spk_segments.append((start_time, end_time))
        if spk_id is None or len(spk_segments) == 0:
            continue
        key = (meet_id, local_id)
        channel = spk_to_channel_map[meet_id][spk_id]
        segments[key] = (spk_id, channel, spk_segments)

    # Now we go through each speaker's word-level annotations and store them
    words = {}
    for file in (transcripts_dir / "Words").glob("*.xml"):
        meet_id, local_id, _ = file.stem.split(".")
        key = (meet_id, local_id)
        if key not in segments:
            continue
        else:
            spk_id, channel, spk_segments = segments[key]

        seg_words = []
        combine_with_next = False
        with open(file) as f:
            tree = ET.parse(f)
            for i, word in enumerate(tree.getroot()):
                if (
                    word.tag != "w"
                    or "starttime" not in word.attrib
                    or word.attrib["starttime"] == ""
                    or "endtime" not in word.attrib
                    or word.attrib["endtime"] == ""
                ):
                    continue
                start_time = float(word.attrib["starttime"])
                end_time = float(word.attrib["endtime"])
                seg_words.append((start_time, end_time, word.text))
        words[key] = (spk_id, channel, seg_words)

    # Now we create segment-level annotations by combining the word-level
    # annotations with the speaker segment times. We also normalize the text
    # (if requested). The annotations is a dict indexed by (meeting_id, spk_id, channel).
    annotations = defaultdict(list)

    for key, (spk_id, channel, spk_segments) in segments.items():
        # Get the words for this speaker
        _, _, spk_words = words[key]
        new_key = (key[0], spk_id, channel)
        # Now iterate over the speaker segments and create segment annotations
        for seg_start, seg_end in spk_segments:
            seg_words = list(
                filter(lambda w: w[0] >= seg_start and w[1] <= seg_end, spk_words)
            )
            if len(seg_words) == 0:
                continue
            start = seg_words[0][0]
            end = seg_words[-1][1]
            word_alignments = []
            for w in seg_words:
                w_start = max(start, round(w[0], ndigits=4))
                w_end = min(end, round(w[1], ndigits=4))
                w_dur = add_durations(w_end, -w_start, sampling_rate=16000)
                w_symbol = normalize_text_ami(w[2], normalize=normalize)
                if len(w_symbol) == 0:
                    continue
                if w_dur <= 0:
                    logging.warning(
                        f"Segment {key[0]}.{spk_id}.{channel} at time {start}-{end} "
                        f"has a word with zero or negative duration. Skipping."
                    )
                    continue
                word_alignments.append(
                    AlignmentItem(start=w_start, duration=w_dur, symbol=w_symbol)
                )
            text = " ".join(w.symbol for w in word_alignments)
            annotations[new_key].append(
                IcsiSegmentAnnotation(
                    text=text,
                    speaker=spk_id,
                    gender=spk_id[0],
                    start_time=start,
                    end_time=end,
                    words=word_alignments,
                )
            )
    return annotations, channel_to_idx_map


# IHM and MDM audio requires grouping multiple channels of AudioSource into
# one Recording.


def prepare_audio_grouped(
    audio_paths: List[Pathlike],
    channel_to_idx_map: Dict[str, Dict[str, int]] = None,
    save_to_wav: bool = False,
    output_dir: Pathlike = None,
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

        if save_to_wav:
            session_dir = Path(output_dir) / "wavs" / session_name
            session_dir.mkdir(parents=True, exist_ok=True)
            for i, audio_path in enumerate(channel_paths):
                audio, _ = read_sph(audio_path)
                wav_path = session_dir / f"{audio_path.stem}.wav"
                sf.write(wav_path, audio.T, samplerate)
                # Replace the sph path with the wav path
                channel_paths[i] = wav_path

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
    save_to_wav: bool = False,
    output_dir: Pathlike = None,
) -> RecordingSet:
    import soundfile as sf

    recordings = []
    for audio_path in tqdm(audio_paths, desc="Preparing audio"):
        session_name = audio_path.parts[-2]
        if audio_path.suffix == ".wav":
            audio_sf = sf.SoundFile(audio_path)
            num_frames = audio_sf.frames
            num_channels = audio_sf.channels
            samplerate = audio_sf.samplerate
        else:
            audio_sf, samplerate = read_sph(audio_path)
            num_channels, num_frames = audio_sf.shape

            if save_to_wav:
                session_dir = Path(output_dir) / "wavs" / session_name
                session_dir.mkdir(parents=True, exist_ok=True)
                wav_path = session_dir / f"{audio_path.stem}.wav"
                sf.write(wav_path, audio_sf.T, samplerate)
                # Replace the sph path with the wav path
                audio_path = wav_path

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
                            alignment={"word": seg_info.words},
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
                        channel=recording.channel_ids,
                        language="English",
                        speaker=seg_info.speaker,
                        gender=seg_info.gender,
                        text=seg_info.text,
                        alignment={"word": seg_info.words},
                    )
                )
    return SupervisionSet.from_segments(segments)


def prepare_icsi(
    audio_dir: Pathlike,
    transcripts_dir: Optional[Pathlike] = None,
    output_dir: Optional[Pathlike] = None,
    mic: Optional[str] = "ihm",
    normalize_text: str = "kaldi",
    save_to_wav: bool = False,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param audio_dir: Pathlike, the path which holds the audio data
    :param transcripts_dir: Pathlike, the path which holds the transcripts data
    :param output_dir: Pathlike, the path where to write the manifests - `None` means manifests aren't stored on disk.
    :param mic: str {'ihm','ihm-mix','sdm','mdm'}, type of mic to use.
    :param normalize_text: str {'none', 'upper', 'kaldi'} normalization of text
    :param save_to_wav: bool, whether to save the sph audio to wav format
    :return: a Dict whose key is ('train', 'dev', 'test'), and the values are dicts of manifests under keys
        'recordings' and 'supervisions'.
    """
    audio_dir = Path(audio_dir)
    transcripts_dir = (
        Path(transcripts_dir)
        if transcripts_dir is not None
        else audio_dir / "transcripts"
    )

    assert audio_dir.is_dir(), f"No such directory: {audio_dir}"
    assert transcripts_dir.is_dir(), f"No such directory: {transcripts_dir}"
    assert mic in MIC_TO_CHANNELS.keys(), f"Mic {mic} not supported"

    if save_to_wav:
        assert output_dir is not None, "output_dir must be specified when saving to wav"

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
            list(audio_paths),
            channel_to_idx_map if mic == "ihm" else None,
            save_to_wav,
            output_dir,
        )
    elif mic == "sdm" or mic == "ihm-mix":
        audio_paths = (
            audio_dir.rglob(f"chan[{channels}].sph")
            if len(channels)
            else audio_dir.rglob("*.wav")
        )
        audio = prepare_audio_single(list(audio_paths), save_to_wav, output_dir)

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

        audio_part, supervision_part = fix_manifests(audio_part, supervision_part)
        validate_recordings_and_supervisions(audio_part, supervision_part)

        # Write to output directory if a path is provided
        if output_dir is not None:
            audio_part.to_file(output_dir / f"icsi-{mic}_recordings_{part}.jsonl.gz")
            supervision_part.to_file(
                output_dir / f"icsi-{mic}_supervisions_{part}.jsonl.gz"
            )

        # Combine all manifests into one dictionary
        manifests[part] = {"recordings": audio_part, "supervisions": supervision_part}

    return dict(manifests)
