"""
Data preparation recipe for CSLU Kids corpus (https://catalog.ldc.upenn.edu/LDC2007S18):

Summary of corpus from LDC webpage:

Collection of spontaneous and prompted speech from 1100 children between Kindergarten 
and Grade 10 in the Forest Grove School District in Oregon. All children -- approximately 
100 children at each grade level -- read approximately 60 items from a total list of 319 
phonetically-balanced but simple words, sentences or digit strings. Each utterance of 
spontaneous speech begins with a recitation of the alphabet and contains a monologue of 
about one minute in duration. This release consists of 1017 files containing approximately 
8-10 minutes of speech per speaker. Corresponding word-level transcriptions are also included.

Prompted speech is verified and divided into following categories:

1 Good: Only the target word is said.
2 Maybe: Target word is present, but there's other junk in the file.
3 Bad: Target word is not said. 
4 Puff: Same as good, but w/ an air puff.

This data is not available for free - your institution needs to have an LDC subscription.
"""
import logging
import re
from pathlib import Path
from typing import Dict, Optional, Union

from tqdm import tqdm

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, check_and_rglob

NOISE_TAGS_REGEX = re.compile("<.*?>")


def read_text(file: Path, normalize: Optional[bool] = True) -> str:
    with open(file, "r") as f:
        text = f.read().replace("\n", " ")
        text = re.sub(NOISE_TAGS_REGEX, "", text) if normalize else text
    return text


def prepare_cslu_kids(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    absolute_paths: Optional[bool] = True,
    normalize_text: Optional[bool] = True,
) -> Dict[str, Union[RecordingSet, SupervisionSet]]:
    """
    Prepare manifests for CSLU Kids corpus. The supervision contains either the
    prompted text, or a transcription of the spontaneous speech, depending on
    whether the utterance was scripted or spontaneous.

    Additionally, the following information is present in the `custom` tag:
    scripted/spontaneous utterance, and verification label (rating between 1 and 4)
    for scripted utterances (see https://catalog.ldc.upenn.edu/docs/LDC2007S18/verification-note.txt
    or top documentation in this script for more information).

    :param corpus_dir: Path to downloaded LDC corpus.
    :param output_dir: Directory where the manifests should be written. Can be omitted to avoid writing.
    :param absolute_paths: Wheter to write absolute paths to audio sources (default = False)
    :param normalize_text: remove noise tags (<bn>, <bs>) from spontaneous speech transcripts (default = True)
    :return: A dict with manifests. The keys are: ``{'recordings', 'supervisions'}``.
    """
    corpus_dir = Path(corpus_dir) if isinstance(corpus_dir, str) else corpus_dir

    # Get list of all recordings
    audio_paths = check_and_rglob(corpus_dir, "*.wav")

    # Read verification labels
    verification = {}
    for file in check_and_rglob(corpus_dir, "*-verified.txt"):
        with open(file, "r") as f:
            for line in f:
                path, label = line.strip().split()
                utt = Path(path).stem
                verification[utt] = int(label)

    # Read prompted transcriptions
    prompts = {}
    with open(corpus_dir / "docs" / "all.map", "r") as f:
        for line in f:
            if line.strip() != "":
                prompt, text = line.strip().split(maxsplit=1)
                prompts[prompt] = text[1:-1]  # remove " " around the text

    recordings = []
    supervisions = []
    for p in tqdm(audio_paths, desc="Preparing manifests"):

        # /data/corpora/LDC2007S18/speech/scripted/00/0/ks001/ks001000.wav
        uttid = p.stem  # ks001000
        spk = p.parent.stem  # ks001
        cat = p.parent.parent.stem  # 0
        prompt = p.parent.parent.parent.stem  # 00
        type = p.parent.parent.parent.parent.stem  # scripted

        recording = Recording.from_file(
            p, relative_path_depth=None if absolute_paths else 3
        )
        recordings.append(recording)

        if type == "scripted":
            text = prompts[prompt]
            verification_label = verification[uttid] if uttid in verification else None
            custom = {"type": type, "verification_label": verification_label}
        elif type == "spontaneous":
            text = read_text(
                corpus_dir / "trans" / type / prompt / cat / spk / f"{uttid}.txt",
                normalize=normalize_text,
            )
            custom = {"type": type}
        supervisions.append(
            SupervisionSegment(
                id=uttid,
                recording_id=uttid,
                start=0,
                duration=recording.duration,
                speaker=spk,
                language="English",
                text=text,
                custom=custom,
            )
        )

    recordings = RecordingSet.from_recordings(recordings)
    supervisions = SupervisionSet.from_segments(supervisions)

    validate_recordings_and_supervisions(recordings, supervisions)

    manifests = {
        "recordings": recordings,
        "supervisions": supervisions,
    }

    if output_dir is not None:
        logging.info("Writing manifests to JSON files")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifests["recordings"].to_json(output_dir / "recordings.json")
        manifests["supervisions"].to_json(output_dir / "supervisions.json")

    return manifests
