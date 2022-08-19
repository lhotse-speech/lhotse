"""
optional TAL_CSASR(587 hours) if available(https://ai.100tal.com/dataset).
It is a mandarin-english code-switch corpus.
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Union

from tqdm.auto import tqdm

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike


def prepare_tal_csasr(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'recordings' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    transcript_path = corpus_dir / "TAL_CSASR" / "label"
    transcript_dict = {}
    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            idx_transcript = line.split()
            transcript_dict[idx_transcript[0]] = " ".join(idx_transcript[1:])

    manifests = defaultdict(dict)
    dataset_parts = ["train"]
    for part in tqdm(
        dataset_parts,
        desc="Process tal_csasr audio, it takes about 4 minutes using 40 cpu jobs.",
    ):
        logging.info(f"Processing tal_csasr subset: {part}")
        # Generate a mapping: utt_id -> (audio_path, audio_info, speaker, text)

        supervisions = []
        wav_path = corpus_dir / "TAL_CSASR" / "cs_wav"
        recordings = RecordingSet.from_dir(
            path=wav_path, pattern="*.wav", num_jobs=num_jobs
        )

        for audio_path in wav_path.rglob("**/*.wav"):

            idx = audio_path.stem
            speaker = idx
            if idx not in transcript_dict:
                logging.warning(f"No transcript: {idx}")
                logging.warning(f"{audio_path} has no transcript.")
                continue
            text = transcript_dict[idx]
            if not audio_path.is_file():
                logging.warning(f"No such file: {audio_path}")
                continue

            segment = SupervisionSegment(
                id=idx,
                recording_id=idx,
                start=0.0,
                duration=recordings.duration(idx),
                channel=0,
                language="Chinese",
                speaker=speaker,
                text=text.strip(),
            )
            supervisions.append(segment)

        recording_set = RecordingSet.from_recordings(recordings)
        supervision_set = SupervisionSet.from_segments(supervisions)
        validate_recordings_and_supervisions(recording_set, supervision_set)

        if output_dir is not None:
            supervision_set.to_file(
                output_dir / f"tal_csasr_supervisions_{part}.jsonl.gz"
            )
            recording_set.to_file(output_dir / f"tal_csasr_recordings_{part}.jsonl.gz")

        manifests[part] = {"recordings": recording_set, "supervisions": supervision_set}

    return manifests
