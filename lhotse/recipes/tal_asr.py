"""
optional TAL_ASR (100 hours) if available(https://ai.100tal.com/dataset).

"""
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Union

from tqdm.auto import tqdm

from lhotse import fix_manifests, validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike


def text_normalize(line: str):
    """
    It is from https://github.com/wenet-e2e/wenet/blob/main/examples/multi_cn/s0/local/tal_data_prep.sh#L57
      sed 's/Ａ/A/g' | sed 's/#//g' | sed 's/=//g' | sed 's/、//g' | \
    sed 's/，//g' | sed 's/？//g' | sed 's/。//g' | sed 's/[ ][ ]*$//g'\
    """
    line = line.replace("Ａ", "A")
    line = re.sub(f"#|=|、|，|？|。|[|]", "", line)
    line = line.upper()
    return line


def prepare_tal_asr(
    corpus_dir: Pathlike, output_dir: Optional[Pathlike] = None
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

    transcript_path = corpus_dir / "aisolution_data" / "transcript" / "transcript.txt"
    transcript_dict = {}
    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            idx_transcript = line.split()
            content = " ".join(idx_transcript[1:])
            content = text_normalize(content)
            transcript_dict[idx_transcript[0]] = content

    manifests = defaultdict(dict)
    dataset_parts = ["train", "dev", "test"]
    for part in tqdm(
        dataset_parts,
        desc="Process tal_asr audio, it takes about 447 seconds.",
    ):
        logging.info(f"Processing tal_asr subset: {part}")
        # Generate a mapping: utt_id -> (audio_path, audio_info, speaker, text)
        recordings = []
        supervisions = []
        wav_path = corpus_dir / "aisolution_data" / "wav" / f"{part}"
        for audio_path in wav_path.rglob("**/*.wav"):

            idx = audio_path.stem
            speaker = audio_path.parts[-2]
            if idx not in transcript_dict:
                logging.warning(f"No transcript: {idx}")
                logging.warning(f"{audio_path} has no transcript.")
                continue
            text = transcript_dict[idx]
            if not audio_path.is_file():
                logging.warning(f"No such file: {audio_path}")
                continue
            recording = Recording.from_file(audio_path)
            recordings.append(recording)
            segment = SupervisionSegment(
                id=idx,
                recording_id=idx,
                start=0.0,
                duration=recording.duration,
                channel=0,
                language="Chinese",
                speaker=speaker,
                text=text.strip(),
            )
            supervisions.append(segment)

        recording_set = RecordingSet.from_recordings(recordings)
        supervision_set = SupervisionSet.from_segments(supervisions)

        recording_set, supervision_set = fix_manifests(recording_set, supervision_set)
        validate_recordings_and_supervisions(recording_set, supervision_set)

        if output_dir is not None:
            supervision_set.to_file(
                output_dir / f"tal_asr_supervisions_{part}.jsonl.gz"
            )
            recording_set.to_file(output_dir / f"tal_asr_recordings_{part}.jsonl.gz")

        manifests[part] = {"recordings": recording_set, "supervisions": supervision_set}

    return manifests
