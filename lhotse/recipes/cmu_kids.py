"""
Data preparation recipe for CMU Kids corpus (https://catalog.ldc.upenn.edu/LDC97S63):

Summary of corpus from LDC webpage:

This database is comprised of sentences read aloud by children. It was originally designed 
in order to create a training set of children's speech for the SPHINX II automatic speech 
recognizer for its use in the LISTEN project at Carnegie Mellon University.

The children range in age from six to eleven (see details below) and were in first through 
third grades (the 11-year-old was in 6th grade) at the time of recording. There were 24 male 
and 52 female speakers. There are 5,180 utterances in all.

The speakers come from two separate populations:

 1. SIM95: They were recorded in the summer of 1995 and were enrolled in either the Chatham 
    College Summer Camp or the Mount Lebanon Extended Day Summer Fun program in Pittsburgh. 
    They were recorded on-site. There are 44 speakers and 3,333 utterances in this set. They
    "good" reading examples.
 2. FP: These are examples of errorful reading and dialectic variants. The readers come from 
    Fort Pitt School in Pittsburgh and were recorded in April 1996. There are 32 speakers and 
    1,847 utterances in this set.

The user should be aware that the speakers' dialect partly reflects what is locally called "Pittsburghese."

The corpus does not come with a train/dev/test split, and the Kaldi recipe splits it randomly
into 70%/30% train-test. We do not perform any splits, and just return the complete recording
and supervision manifests.

This data is not available for free - your institution needs to have an LDC subscription.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike


def prepare_cmu_kids(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    absolute_paths: Optional[bool] = True,
) -> Dict[str, Union[RecordingSet, SupervisionSet]]:
    """
    Prepare manifests for CMU Kids corpus. The prepared supervisions contain the
    prompt text as the `text`. Additionally, in the `custom` tag, we provide the
    following data: speaker grade/age, population where the speaker came from
    (SIM95/FP), spoken transcript, and transcription bin (1/2).

    Here, bin `1` means utterances where the speaker followed the prompt and no
    noise/mispronunciation is present, and `2` refers to noisy utterances.

    The tag `spoken_transcript` is the transcription that was actually spoken. It
    contains noise tags and phone transcription in case the pronunciation differed
    from that in CMU Dict.

    :param corpus_dir: Path to downloaded LDC corpus.
    :param output_dir: Directory where the manifests should be written. Can be omitted to avoid writing.
    :param absolute_paths: Wheter to write absolute paths to audio sources (default = False)
    :return: A dict with manifests. The keys are: ``{'recordings', 'supervisions'}``.
    """
    corpus_dir = Path(corpus_dir) if isinstance(corpus_dir, str) else corpus_dir
    corpus_dir = corpus_dir.parent if corpus_dir.stem == "cmu_kids" else corpus_dir

    recordings = []
    supervisions = []

    # Get transcripts for all utterances
    utterances = {}
    with open(corpus_dir / "cmu_kids" / "tables" / "sentence.tbl", "r") as f:
        for line in f:
            utt, count, text = line.strip().split("\t")
            utterances[utt] = text

    # Get speaker metadata
    speaker_info = {}
    with open(corpus_dir / "cmu_kids" / "tables" / "speaker.tbl", "r") as f:
        for _ in range(2):
            next(f)
        for line in f:
            # ID    LOC     GR/AGE  TOT     BIN2
            # fabm    SUM95   3/9     100     62
            # facs    SUM95   2/8     90      55
            spk, pop, gr_age, _, _ = line.strip().split("\t")
            grade, age = gr_age.split("/")
            speaker_info[spk] = (pop, grade, age)

    # Iterate through all transcriptions and add to supervisions
    with open(corpus_dir / "cmu_kids" / "tables" / "transcrp.tbl", "r") as f:
        for line in f:
            trn_id, transcript = line.strip().split(maxsplit=1)
            spk = trn_id[0:4]
            utt = trn_id[4:7]
            bin = int(trn_id[7])
            pop, grade, age = speaker_info[spk]

            audio_path = (
                corpus_dir / "cmu_kids" / "kids" / spk / "signal" / f"{trn_id}.sph"
            )
            recording = Recording.from_file(
                audio_path, relative_path_depth=None if absolute_paths else 3
            )
            recordings.append(recording)

            supervisions.append(
                SupervisionSegment(
                    id=trn_id,
                    recording_id=trn_id,
                    start=0,
                    duration=recording.duration,
                    speaker=spk,
                    gender="Male" if spk[0] == "m" else "Female",
                    language="English",
                    text=utterances[utt],
                    custom={
                        "speaker_grade": grade if grade != "NA" else None,
                        "speaker_age": int(age) if age != "NA" else None,
                        "speaker_population": pop,
                        "bin": bin,
                        "spoken_transcript": transcript,
                    },
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
