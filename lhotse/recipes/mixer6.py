"""
This is a data preparation script for the Mixer 6 dataset. The following description
is taken from the LDC website:

Mixer 6 Speech comprises 15,863 hours of audio recordings of interviews, transcript 
readings and conversational telephone speech involving 594 distinct native English 
speakers. This material was collected by LDC in 2009 and 2010 as part of the Mixer 
project, specifically phase 6, the focus of which was on native American English 
speakers local to the Philadelphia area.

The telephone collection protocol was similar to other LDC telephone studies (e.g., 
Switchboard-2 Phase III Audio - LDC2002S06): recruited speakers were connected through 
a robot operator to carry on casual conversations lasting up to 10 minutes, usually 
about a daily topic announced by the robot operator at the start of the call. The raw 
digital audio content for each call side was captured as a separate channel, and each 
full conversation was presented as a 2-channel interleaved audio file, with 8000 
samples/second and u-law sample encoding. Each speaker was asked to complete 15 calls.

The multi-microphone portion of the collection utilized 14 distinct microphones 
installed identically in two mutli-channel audio recording rooms at LDC. Each session 
was guided by collection staff using prompting and recording software to conduct the 
following activities: (1) repeat questions (less than one minute), (2) informal 
conversation (typically 15 minutes), (3) transcript reading (approximately 15 minutes) 
and (4) telephone call (generally 10 minutes). Speakers recorded up to three 45-minute 
sessions on distinct days. The 14 channels were recorded synchronously into separate 
single-channel files, using 16-bit PCM sample encoding at 16000 samples/second.

The collection contains 4,410 recordings made via the public telephone network and 
1,425 sessions of multiple microphone recordings in office-room settings. The telephone 
recordings are presented as 8-KHz 2-channel NIST SPHERE files, and the microphone 
recordings are 16-KHz 1-channel flac/ms-wav files. 
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Union, NamedTuple

from tqdm import tqdm

from lhotse import validate_recordings_and_supervisions, fix_manifests
from lhotse.audio import Recording, RecordingSet, AudioSource, sph_info
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, Seconds, is_module_available


class MixerSegmentAnnotation(NamedTuple):
    session: str
    speaker: str
    start: Seconds
    end: Seconds
    text: str


def prepare_mixer6(
    corpus_dir: Pathlike,
    transcript_dir: Optional[Pathlike],
    output_dir: Optional[Pathlike] = None,
    part: str = "intv",
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Pathlike, the path of the speech corpus dir (LDC2013S03).
    :param transcript_dir: Pathlike, the path of the transcript dir (Mixer6Transcription).
    :param output_dir: Pathlike, the path where to write the manifests.
    :param part: str, "call" or "intv", specifies whether to prepare the telephone or interview data.
    :return: a Dict whose key is the dataset part ('dev' and 'dev_test'), and the value is Dicts with the keys 'recordings' and 'supervisions'.

    NOTE on interview data: each recording in the interview data contains 14 channels. Channel 0 (Mic 01)
    is the lapel mic for the interviewer, so it can be treated as close-talk. Channel 1 (Mic 02) is
    the lapel mic for the interviewee.  All other mics are placed throughout the room.
    Channels 2 and 13 (Mics 03 and 14) are often silent, and so they may be removed.

    NOTE: the official LDC corpus does not contain transcriptions for the data.
    """
    if not is_module_available("textgrid"):
        raise ValueError(
            "To prepare Mixer 6 data, please 'pip install textgrid' first."
        )
    import textgrid

    corpus_dir = Path(corpus_dir)
    corpus_dir = (
        corpus_dir / "mx6_speech" if corpus_dir.stem != "mx6_speech" else corpus_dir
    )
    transcript_dir = Path(transcript_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    assert part in ["call", "intv"]

    manifests = defaultdict(dict)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Read speaker gender info
    spk_to_gender = {}
    with open(corpus_dir / "docs" / "mx6_subjs.csv", "r") as f:
        next(f)
        for line in f:
            parts = line.strip().split(",")
            spk_to_gender[parts[0]] = parts[1]

    recordings = []
    supervisions = []

    if part == "call":
        text_paths = transcript_dir.rglob("*/call/*.textgrid")
        for text_path in tqdm(text_paths, desc="Processing call textgrids"):
            session_id = "_".join(text_path.stem.split("_")[:-2])
            speaker_id = session_id.split("_")[-1]
            tg = textgrid.TextGrid.fromFile(str(text_path))
            for i in range(len(tg.tiers)):
                for j in range(len(tg.tiers[i].intervals)):
                    if tg.tiers[i].intervals[j].mark != "":
                        start = tg.tiers[i].intervals[j].minTime
                        end = tg.tiers[i].intervals[j].maxTime
                        text = " ".join(tg.tiers[i].intervals[j].mark.split(" ")[1:])
                        segment = SupervisionSegment(
                            id=f"{session_id}-{i}-{j}",
                            recording_id=session_id,
                            start=start,
                            duration=round(end - start, 4),
                            channel=0,
                            language="English",
                            speaker=f"{speaker_id}-{i}",
                            text=text,
                        )
                        supervisions.append(segment)
            for path in tqdm(
                list(corpus_dir.rglob("*.sph")), desc="Processing call recordings"
            ):
                recordings.append(Recording.from_file(path))

    else:
        import soundfile as sf

        intv_list = {}
        with open(corpus_dir / "docs" / "mx6_intvs.csv", "r") as f:
            next(f)
            for line in f:
                parts = line.strip().split(",")
                spk_id = parts[0]
                session_time = parts[1].split("_")[0]
                intv_list[f"{spk_id}_{session_time}"] = parts[1]

        text_paths = list(transcript_dir.rglob(f"intv/*.textgrid"))
        for text_path in tqdm(text_paths, desc="Processing intv textgrids"):
            try:
                spk_id, session_time, _, _ = text_path.stem.split("_")
            except ValueError:
                logging.warning(f"Skipping {text_path.stem}")
                continue
            intv = f"{spk_id}_{session_time}"
            audio_id = intv_list[intv]
            tg = textgrid.TextGrid.fromFile(str(text_path))
            for i, tier in enumerate(tg.tiers):
                for j, interval in enumerate(tier.intervals):
                    if interval.mark != "":
                        start = interval.minTime
                        end = interval.maxTime
                        text = " ".join(interval.mark.split(" ")[1:])
                        segment = SupervisionSegment(
                            id=f"{intv}-{i}-{j}",
                            recording_id=audio_id,
                            start=start,
                            duration=round(end - start, 4),
                            channel=0,
                            language="English",
                            speaker=f"{spk_id}-{i}",
                            text=text,
                        )
                        supervisions.append(segment)

            audios = list(corpus_dir.rglob(f"{audio_id}*.flac"))
            if len(audios) == 0:
                logging.warning(f"No audio for {audio_id}")
                continue
            audio_sf = sf.SoundFile(str(audios[0]))
            recordings.append(
                Recording(
                    id=audio_id,
                    sources=[
                        AudioSource(
                            type="file",
                            channels=[int(audio.stem[-2:]) - 1],
                            source=str(audio),
                        )
                        for audio in sorted(audios)
                    ],
                    sampling_rate=audio_sf.samplerate,
                    num_samples=audio_sf.frames,
                    duration=audio_sf.frames / audio_sf.samplerate,
                )
            )

    recording_set = RecordingSet.from_recordings(recordings)
    supervision_set = SupervisionSet.from_segments(supervisions)

    recording_set, supervision_set = fix_manifests(recording_set, supervision_set)
    validate_recordings_and_supervisions(recording_set, supervision_set)

    if output_dir is not None:
        recording_set.to_file(output_dir / f"recordings.jsonl")
        supervision_set.to_file(output_dir / f"supervisions.jsonl")

    manifests = {"recordings": recording_set, "supervisions": supervision_set}

    return manifests
