"""
 About the eval2000 corpus
     2000 HUB5 English Evaluation was developed by the Linguistic Data Consortium (LDC) and
     consists of approximately 11 hours of English conversational telephone speech used in the
     2000 HUB5 evaluation sponsored by NIST (National Institute of Standards and Technology).
     The source data consists of conversational telephone speech collected by LDC:
     (1) 20 unreleased telephone conversations from the Swtichboard studies in which recruited
      speakers were connected through a robot operator to carry on casual conversations about a
      daily topic announced by the robot operator at the start of the call; and
     (2) 20 telephone conversations from CALLHOME American English Speech which consists of
      unscripted telephone conversations between native English speakers.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from lhotse.audio import Recording, RecordingSet
from lhotse.qa import fix_manifests, validate_recordings_and_supervisions
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, check_and_rglob

EVAL2000_AUDIO_DIR = "LDC2002S09"
EVAL2000_TRANSCRIPT_DIR = "LDC2002T43"


def prepare_eval2000(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    transcript_path: Optional[Pathlike] = None,
    absolute_paths: bool = False,
    num_jobs: int = 1,
) -> Dict[str, Union[RecordingSet, SupervisionSet]]:
    """
    Prepares manifests for Eval2000.

    :param corpus_path: Path to global corpus
    :param output_dir: Directory where the manifests should be written. Can be omitted to avoid writing.
    :param absolute_paths: Whether to return absolute or relative (to the corpus dir) paths for recordings.
    :return: A dict with manifests. The keys are: ``{'recordings', 'supervisions'}``.
    """

    corpus_dir = Path(corpus_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_partition_dir_path = corpus_dir / EVAL2000_AUDIO_DIR / "hub5e_00" / "english"
    assert (
        audio_partition_dir_path.is_dir()
    ), f"No such directory:{audio_partition_dir_path}"
    default_transcript_path = (
        corpus_dir / EVAL2000_TRANSCRIPT_DIR / "reference" / "english"
    )
    transcript_dir_path = (
        default_transcript_path if transcript_path is None else Path(transcript_path)
    )
    assert transcript_dir_path.is_dir(), f"No such directory:{transcript_dir_path}"
    groups = []
    for path in (audio_partition_dir_path).rglob("*.sph"):
        base = Path(path).stem
        groups.append({"audio": path})
    recordings = RecordingSet.from_recordings(
        Recording.from_file(
            group["audio"], relative_path_depth=None if absolute_paths else 3
        )
        for group in groups
    )
    segments = make_segments(transcript_dir_path)
    supervisions = SupervisionSet.from_segments(segments)
    recordings, supervisions = fix_manifests(recordings, supervisions)
    validate_recordings_and_supervisions(recordings, supervisions)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        recordings.to_file(output_dir / "eval2000_recordings_all.jsonl.gz")
        supervisions.to_file(output_dir / "eval2000_supervisions_unnorm.jsonl.gz")
    return {"recordings": recordings, "supervisions": supervisions}


def make_segments(transcript_dir_path, omit_silence: bool = True):
    segment_supervision = []
    for text_path in (transcript_dir_path).rglob("*.txt"):
        trans_file = Path(text_path).stem
        trans_file_lines = [l.split() for l in open(text_path)]
        id = -1
        for i in range(0, len(trans_file_lines)):
            if trans_file_lines[i]:  # skip empty lines
                trans_line = trans_file_lines[i]  # ref line
                if "#" not in trans_line[0]:  # skip header lines of the file
                    id = id + 1
                    start = float(trans_line[0])
                    end = float(trans_line[1])
                    duration = round(end - start, ndigits=8)
                    side = (trans_line[2].split(":"))[0]
                    if side == "A":
                        channel = 0
                    else:
                        channel = 1
                    text_line = " ".join(trans_line[3::])
                    segment_id = trans_file + "-" + str(id)
                    recording_id = trans_file
                    speaker = trans_file + "-" + side
                    segment = SupervisionSegment(
                        id=segment_id,
                        recording_id=recording_id,
                        start=start,
                        duration=duration,
                        channel=channel,
                        language="English",
                        speaker=speaker,
                        text=text_line,
                    )
                    segment_supervision.append(segment)
    return segment_supervision
    # transcript lines  in one .txt file looks like this
    """
    #Language: eng
    #File id: 5017
    #Starting at 121 Ending at 421
    # 121 131 #BEGIN
    # 411 421 #END

    116.17 121.98 A: <contraction e_form="[we=>we]['re=>are]">we're starting the transition I you know told the students that they were going to you know, what the new plan was and

    121.79 122.43 B: mhm

    122.93 126.57 A: %um, <contraction e_form="[they=>they]['re=>are]">they're not that thrilled about it, but %uh

    126.30 128.83 B: what to you mean? {breath} oh, about <contraction e_form="[you=>you]['re=>are]">you're leaving?
    """
