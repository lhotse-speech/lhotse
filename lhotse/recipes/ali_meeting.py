"""
This recipe prepares the AliMeeting corpus. It consists of 120 hours of real recorded 
Mandarin meeting data, including far-field data collected by an 8-channel microphone 
array as well as near-field data collected by each participant's headset microphone.

This corpus isn't publicly released yet. A pre-release version has been made
available for CLSP use at /export/c01/corpora6/AliMeeting, courtesy of Hang Lv.
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Union

from tqdm import tqdm

from lhotse import validate_recordings_and_supervisions, fix_manifests
from lhotse.audio import AudioSource, Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, is_module_available


def prepare_ali_meeting(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    mic: Optional[str] = "far",
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param mic: str, "near" or "far", specifies whether to prepare the near-field or far-field data.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'recordings' and 'supervisions'.
    """
    if not is_module_available("textgrid"):
        raise ValueError(
            "To prepare AliMeeting data, please 'pip install textgrid' first."
        )
    import textgrid

    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    manifests = defaultdict(dict)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    for part in ["Train", "Eval", "Test"]:
        recordings = []
        supervisions = []
        wav_paths = corpus_dir / f"{part}_Ali_{mic}" / "audio_dir"
        text_paths = corpus_dir / f"{part}_Ali_{mic}" / "textgrid_dir"

        # For 'near' setting:
        #  - wav files have names like R0003_M0046_F_SPK0093.wav
        #  - textgrid files have names like R0003_M0046_F_SPK0093.TextGrid
        # Speaker ID information is present in the file name itself

        # For 'far' setting:
        #  - wav files have names like R0015_M0151_MS002.wav
        #  - textgrid files have names like R0015_M015.TextGrid
        # Speaker ID information is present inside the TextGrid file

        for text_path in tqdm(
            list(text_paths.rglob("*.TextGrid")), desc=f"Preparing {part}"
        ):
            session_id = text_path.stem

            if mic == "near":
                _, _, gender, spk_id = session_id.split("_")
                spk_id = spk_id[3:]  # SPK1953 -> 1953

            try:
                tg = textgrid.TextGrid.fromFile(str(text_path))
            except ValueError:
                logging.warning(
                    f"{session_id} has annotation issues. Skipping this recording."
                )
                continue

            wav_path = list(wav_paths.rglob(f"{session_id}*.wav"))[0]

            recording = Recording.from_file(wav_path, recording_id=session_id)
            recordings.append(recording)

            for tier in tg.tiers:
                if mic == "far":
                    parts = tier.name.split("_")
                    if len(parts) == 4:
                        _, _, gender, spk_id = parts
                    elif len(parts) == 2:
                        gender, spk_id = parts
                    spk_id = spk_id[3:]  # SPK1953 -> 1953

                for i, interval in enumerate(tier.intervals):
                    if interval.mark != "":
                        start = interval.minTime
                        end = interval.maxTime
                        text = interval.mark
                        segment = SupervisionSegment(
                            id=f"{session_id}-{spk_id}-{i}",
                            recording_id=recording.id,
                            start=start,
                            duration=round(end - start, 4),
                            channel=0,
                            language="Chinese",
                            speaker=spk_id,
                            gender=gender,
                            text=text.strip(),
                        )
                        supervisions.append(segment)

        recording_set, supervision_set = fix_manifests(
            RecordingSet.from_recordings(recordings),
            SupervisionSet.from_segments(supervisions),
        )
        # Fix manifests
        validate_recordings_and_supervisions(recording_set, supervision_set)

        if output_dir is not None:
            supervision_set.to_file(output_dir / f"supervisions_{part.lower()}.jsonl")
            recording_set.to_file(output_dir / f"recordings_{part.lower()}.jsonl")

        manifests[part.lower()] = {
            "recordings": recording_set,
            "supervisions": supervision_set,
        }

    return manifests
