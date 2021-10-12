"""
About the Fisher Spanish corpus

    This is conversational telephone speech collected as 2-channel μ-law, 8kHz-sampled data. 
    The catalog number LDC2010S01 for audio corpus and LDC2010T04 for transcripts.

    This data is not available for free - your institution needs to have an LDC subscription.
"""

import codecs
import itertools as it
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from tqdm.auto import tqdm

from lhotse.audio import RecordingSet
from lhotse.qa import fix_manifests, validate_recordings_and_supervisions
from lhotse.recipes.fisher_english import create_recording
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, check_and_rglob


def create_supervision(
    sessions_and_transcript_path: Tuple[Dict[str, Dict[str, str]], Pathlike]
) -> List[SupervisionSegment]:

    sessions, transcript_path = sessions_and_transcript_path
    transcript_path = Path(transcript_path)
    with codecs.open(transcript_path, "r", "utf8") as trans_f:

        lines = [l.rstrip("\n") for l in trans_f.readlines()][3:]
        lines = [l.split("\t") for l in lines if l.strip() != ""]
        lines = [
            [
                float(l[2]),
                float(l[3]),
                int(l[1]),
                " ".join([w for w in l[7].split() if w.strip() != ""]),
            ]
            for l in lines
        ]

        segments = [
            SupervisionSegment(
                id=transcript_path.stem + "-" + str(k).zfill(len(str(len(lines)))),
                recording_id=transcript_path.stem,
                start=round(l[0], 10),
                duration=round(l[1] - l[0], 10),
                channel=l[2],
                text=l[3],
                language="Spanish",
                speaker=sessions[transcript_path.stem.split("_")[2]][l[2]],
            )
            for k, l in enumerate(lines)
        ]

    return segments


def prepare_fisher_spanish(
    audio_dir_path: Pathlike,
    transcript_dir_path: Pathlike,
    output_dir: Optional[Pathlike] = None,
    absolute_paths: bool = False,
) -> Dict[str, Union[RecordingSet, SupervisionSet]]:

    """
    Prepares manifests for Fisher Spanish.
    We create two manifests: one with recordings, and the other one with text supervisions.

    :param audio_dir_path: Path to audio directory (usually LDC2010S01).
    :param transcript_dir_path: Path to transcript directory (usually LDC2010T04).
    :param output_dir: Directory where the manifests should be written. Can be omitted to avoid writing.
    :param absolute_paths: Whether to return absolute or relative (to the corpus dir) paths for recordings.
    :return: A dict with manifests. The keys are: ``{'recordings', 'supervisions'}``.
    """

    audio_dir_path, transcript_dir_path = Path(audio_dir_path), Path(
        transcript_dir_path
    )

    audio_paths = check_and_rglob(audio_dir_path, "*.sph")
    transcript_paths = check_and_rglob(transcript_dir_path, "*.tdf")

    sessions_data_path = check_and_rglob(transcript_dir_path, "*_call.tbl")[0]
    with codecs.open(sessions_data_path, "r", "utf8") as sessions_data_f:
        session_lines = [
            l.rstrip("\n").split(",") for l in sessions_data_f.readlines()
        ][1:]
        sessions = {l[0]: {0: l[2], 1: l[8]} for l in session_lines}

    assert len(transcript_paths) == len(sessions) == len(audio_paths)

    create_recordings_input = [(p, None if absolute_paths else 4) for p in audio_paths]
    recordings = [None] * len(audio_paths)
    with ThreadPoolExecutor(os.cpu_count() * 4) as executor:
        with tqdm(total=len(audio_paths), desc="Collect recordings") as pbar:
            for i, reco in enumerate(
                executor.map(create_recording, create_recordings_input)
            ):
                recordings[i] = reco
                pbar.update()
    recordings = RecordingSet.from_recordings(recordings)

    create_supervisions_input = [(sessions, p) for p in transcript_paths]
    supervisions = [None] * len(create_supervisions_input)
    with ThreadPoolExecutor(os.cpu_count() * 4) as executor:
        with tqdm(
            total=len(create_supervisions_input), desc="Create supervisions"
        ) as pbar:
            for i, tmp_supervisions in enumerate(
                executor.map(create_supervision, create_supervisions_input)
            ):
                supervisions[i] = tmp_supervisions
                pbar.update()
    supervisions = list(it.chain.from_iterable(supervisions))
    supervisions = SupervisionSet.from_segments(supervisions).filter(
        lambda s: s.duration > 0.0
    )

    recordings, supervisions = fix_manifests(recordings, supervisions)
    validate_recordings_and_supervisions(recordings, supervisions)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        recordings.to_json(output_dir / "recordings.json")
        supervisions.to_json(output_dir / "supervisions.json")

    return {"recordings": recordings, "supervisions": supervisions}
