"""
This is a data preparation script for the ASpIRE dataset. The following description
is taken from the LDC website:

ASpIRE Development and Development Test Sets was developed for the Automatic Speech 
recognition In Reverberant Environments (ASpIRE) Challenge sponsored by IARPA 
(the Intelligent Advanced Research Projects Activity). It contains approximately 226 
hours of English speech with transcripts and scoring files.

The ASpIRE challenge asked solvers to develop innovative speech recognition systems 
that could be trained on conversational telephone speech, and yet work well on far-
field microphone data from noisy, reverberant rooms. Participants had the opportunity 
to evaluate their techniques on a common set of challenging data that included 
significant room noise and reverberation.

The data is provided in LDC catalog LDC2017S21. The audio data is a subset of Mixer 6 
Speech (LDC2013S03), audio recordings of interviews, transcript readings and 
conversational telephone speech collected by the Linguistic Data Consortium in 2009 
and 2010 from native English speakers local to the Philadelphia area. The transcripts 
were developed by Appen for the ASpIRE challenge.

Data is divided into development and development test sets.

There are 2 versions: "single" and "multi", which stand for single-channel and 
multi-channel audio respectively. All audio is presented as single channel, 16kHz 
16-bit Signed Integer PCM *.wav files. Transcripts are plain text tdf files or as STM
files. Scoring files (glm) are also included.
"""

import logging
import itertools
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Union, NamedTuple

from lhotse import validate_recordings_and_supervisions, fix_manifests
from lhotse.audio import Recording, RecordingSet, AudioSource
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, Seconds


class AspireSegmentAnnotation(NamedTuple):
    session: str
    speaker: str
    start: Seconds
    end: Seconds
    text: str


def prepare_aspire(
    corpus_dir: Pathlike, output_dir: Optional[Pathlike] = None, mic: str = "single"
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Pathlike, the path of the corpus dir (LDC2017S21).
    :param output_dir: Pathlike, the path where to write the manifests.
    :param mic: str, the microphone type, either "single" or "multi".
    :return: a Dict whose key is the dataset part ('dev' and 'dev_test'), and the value is Dicts with the keys 'recordings' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    assert mic in [
        "single",
        "multi",
    ], f"mic must be either 'single' or 'multi', got {mic}"
    corpus_dir = corpus_dir / "IARPA-ASpIRE-Dev-Sets-v2.0" / "data"
    audio_dir = corpus_dir / "dev_and_dev_test_audio"
    stm_dir = corpus_dir / "dev_and_dev_test_STM_files"

    if mic == "single":
        audio_paths = {
            "dev": audio_dir / "ASpIRE_single_dev",
            "dev_test": audio_dir / "ASpIRE_single_dev_test",
        }
        stm_file = {
            "dev": stm_dir / "dev.stm",
            "dev_test": stm_dir / "dev_test.stm",
        }
    else:
        audio_paths = {
            "dev": audio_dir / "ASpIRE_multi_dev",
            "dev_test": audio_dir / "ASpIRE_multi_dev_test",
        }
        stm_file = {
            "dev": stm_dir / "multi_dev.stm",
            "dev_test": stm_dir / "multi_dev_test.stm",
        }
    manifests = defaultdict(dict)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    for part in ["dev", "dev_test"]:
        recordings = []
        supervisions = []

        # Prepare the recordings
        if mic == "single":
            recording_set = RecordingSet.from_dir(audio_paths[part], "*.wav")
        else:
            import soundfile as sf

            audio_groups = {
                k: list(v)
                for k, v in itertools.groupby(
                    sorted(audio_paths[part].glob("*.wav")),
                    key=lambda x: "_".join(x.stem.split("_")[:-1]),
                )
            }  # group audios so that each entry is a session containing all channels
            for session_name, audios in audio_groups.items():
                audio_sf = sf.SoundFile(str(audios[0]))
                recordings.append(
                    Recording(
                        id=session_name,
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

        # Read STM file and prepare segments
        segments = []
        with open(stm_file[part]) as f:
            for line in f:
                session, _, speaker, start, end, text = line.strip().split(maxsplit=5)
                segments.append(
                    AspireSegmentAnnotation(
                        session, speaker, float(start), float(end), text
                    )
                )

        # Group the segments by session and speaker
        segments_grouped = defaultdict(list)
        for segment in segments:
            segments_grouped[(segment.session, segment.speaker)].append(segment)

        # Create the supervisions
        supervisions = []
        for k, segs in segments_grouped.items():
            session, speaker = k
            supervisions += [
                SupervisionSegment(
                    id=f"{session}-{speaker}-{i:03d}",
                    recording_id=session,
                    start=seg.start,
                    duration=round(seg.end - seg.start, 4),
                    speaker=speaker,
                    text=seg.text,
                    language="English",
                )
                for i, seg in enumerate(segs)
            ]
        supervision_set = SupervisionSet.from_segments(supervisions)

        recording_set, supervision_set = fix_manifests(recording_set, supervision_set)
        validate_recordings_and_supervisions(recording_set, supervision_set)

        if output_dir is not None:
            supervision_set.to_file(output_dir / f"supervisions_{part}.jsonl")
            recording_set.to_file(output_dir / f"recordings_{part}.jsonl")

        manifests[part] = {"recordings": recording_set, "supervisions": supervision_set}

    return manifests
