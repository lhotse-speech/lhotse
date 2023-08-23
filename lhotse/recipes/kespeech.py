"""
The KeSpeech is an open source speech dataset, KeSpeech, which involves 1,542 hours of speech
signals recorded by 27,237 speakers in 34 cities in China, and the pronunciation includes
standard Mandarin and its 8 subdialects. The new dataset possesses several properties.
The dataset provides multiple labels including content transcription, speaker identity and
subdialect, hence supporting a variety of speech processing tasks, such as speech recognition,
speaker recognition, and subdialect identification, as well as other advanced techniques
like multi-task learning and conditional learning.

Full paper: https://openreview.net/forum?id=b3Zoeq2sCLq
"""

import logging
from collections import defaultdict
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from tqdm.auto import tqdm

from lhotse.audio import AudioSource, Recording, RecordingSet, info
from lhotse.qa import fix_manifests, validate_recordings_and_supervisions
from lhotse.recipes.utils import manifests_exist
from lhotse.serialization import load_jsonl
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, compute_num_samples

KE_SPEECH_PARTS = ("train_phase1", "train_phase2", "dev_phase1", "dev_phase2", "test")


def text_normalize(line: str) -> str:
    line = line.replace("<SPOKEN_NOISE>", "")
    return line


def prepare_kespeech(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike],
    dataset_parts: Union[str, Sequence[str]] = "all",
    num_jobs: int = 1,
):
    corpus_dir = Path(corpus_dir)
    tasks_dir = corpus_dir / "Tasks" / "ASR"

    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    assert tasks_dir.is_dir(), f"No such directory: {tasks_dir}"

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    subsets = KE_SPEECH_PARTS if "all" in dataset_parts else dataset_parts
    manifests = defaultdict(dict)
    for sub in subsets:
        if sub not in KE_SPEECH_PARTS:
            raise ValueError(f"No such part of dataset in KeSpeech : {sub}")
        manifests[sub] = {"recordings": [], "supervisions": []}

    with ThreadPoolExecutor(num_jobs) as ex:
        for part in tqdm(subsets, desc="Processing KeSpeech", unit="subset"):
            logging.info(f"Processing KeSpeech subset: {part}")

            if manifests_exist(part=part, output_dir=output_dir):
                logging.info(f"KeSpeech subset: {part} already prepared - skipping.")
                continue

            recordings = []
            supervisions = []
            part_path = tasks_dir / part
            futures = []

            with open(file=part_path / "wav.scp") as wav_scp, open(
                file=part_path / "text"
            ) as text, open(file=part_path / "utt2subdialect") as utt2subdialect, open(
                file=part_path / "utt2spk"
            ) as utt2spk:
                for wav_line, text_line, dialect_line, spk_line in zip(
                    wav_scp, text, utt2subdialect, utt2spk
                ):
                    futures.append(
                        ex.submit(
                            parse_utterance,
                            corpus_dir,
                            wav_line,
                            text_line,
                            dialect_line,
                            spk_line,
                        )
                    )
            for future in tqdm(futures, desc="Processing", leave=False):
                result = future.result()
                if result is None:
                    continue
                recording, segment = result
                recordings.append(recording)
                supervisions.append(segment)

            recording_set = RecordingSet.from_recordings(recordings)
            supervision_set = SupervisionSet.from_segments(supervisions)

            # Fix manifests
            recording_set, supervision_set = fix_manifests(
                recording_set, supervision_set
            )
            validate_recordings_and_supervisions(recording_set, supervision_set)

            if output_dir is not None:
                supervision_set.to_file(
                    output_dir / f"kespeech-asr_supervisions_{part}.jsonl.gz"
                )
                recording_set.to_file(
                    output_dir / f"kespeech-asr_recordings_{part}.jsonl.gz"
                )

            manifests[part] = {
                "recordings": recording_set,
                "supervisions": supervision_set,
            }
    return manifests


def parse_utterance(
    corpus_dir: Path,
    wav_line: str,
    text_line: str,
    dialect_line: str,
    spk_line: str,
) -> Optional[Tuple[Recording, SupervisionSegment]]:
    wav_id, wav_path = wav_line.strip().split(maxsplit=1)
    t_wav_id, transcript = text_line.strip().split(maxsplit=1)
    d_wav_id, dialect = dialect_line.strip().split(maxsplit=1)
    s_wav_id, speaker = spk_line.strip().split(maxsplit=1)
    assert wav_id == t_wav_id and t_wav_id == d_wav_id and d_wav_id == s_wav_id
    sampling_rate = 16000
    recording_info = info(corpus_dir / wav_path)
    recording = Recording(
        id=wav_id,
        sources=[
            AudioSource(
                type="file",
                channels=[0],
                source=str(corpus_dir / wav_path),
            )
        ],
        sampling_rate=sampling_rate,
        num_samples=compute_num_samples(
            duration=recording_info.duration,
            sampling_rate=sampling_rate,
        ),
        duration=recording_info.duration,
    )
    supervision = SupervisionSegment(
        id=wav_id,
        recording_id=wav_id,
        start=0.0,
        duration=recording_info.duration,
        text=text_normalize(transcript.strip()),
        language=dialect,
        speaker=speaker,
    )
    return recording, supervision
