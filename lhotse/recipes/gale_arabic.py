"""
The GALE Arabic Broadcast corpus consists of the following LDC corpora:

GALE Arabic phase 2 Conversation Speech
LDC2013S02: http://catalog.ldc.upenn.edu/LDC2013S02
LDC2013S07: http://catalog.ldc.upenn.edu/LDC2013S07
LDC2013T17: http://catalog.ldc.upenn.edu/LDC2013T17
LDC2013T04: http://catalog.ldc.upenn.edu/LDC2013T04

# GALE Arabic phase 2 News Speech
LDC2014S07: http://catalog.ldc.upenn.edu/LDC2014S07
LDC2015S01: http://catalog.ldc.upenn.edu/LDC2015S01
LDC2014T17: http://catalog.ldc.upenn.edu/LDC2014T17
LDC2015T01: http://catalog.ldc.upenn.edu/LDC2015T01

# GALE Arabic phase 3 Conversation Speech
LDC2015S11: http://catalog.ldc.upenn.edu/LDC2015S11
LDC2016S01: http://catalog.ldc.upenn.edu/LDC2016S01
LDC2015T16: http://catalog.ldc.upenn.edu/LDC2015T16
LDC2016T06: http://catalog.ldc.upenn.edu/LDC2016T06

# GALE Arabic phase 3 News Speech
LDC2016S07: http://catalog.ldc.upenn.edu/LDC2016S07
LDC2017S02: http://catalog.ldc.upenn.edu/LDC2017S02
LDC2016T17: http://catalog.ldc.upenn.edu/LDC2016T17
LDC2017T04: http://catalog.ldc.upenn.edu/LDC2017T04

# GALE Arabic phase 4 Conversation Speech
LDC2017S15: http://catalog.ldc.upenn.edu/LDC2017S15
LDC2017T12: http://catalog.ldc.upenn.edu/LDC2017T12

# GALE Arabic phase 4 News Speech
LDC2018S05: http://catalog.ldc.upenn.edu/LDC2018S05
LDC2018T14: http://catalog.ldc.upenn.edu/LDC2018T14

# Training: 941h Testing: 10.4h

The data has two types of speech: conversational and report.
There is no separate dev set provided with the corpus.

The `S` corpora contain speech data and the `T` corpora contain the corresponding
transcriptions. This recipe prepares any subset of these corpora provided as
arguments, but pairs of speech and transcript corpora must be present. E.g.
to only prepare phase 3 news speech, the arguments 
`audio_dirs = ["/export/data/LDC2016S07","/export/data/LDC2017S02"]` and 
`transcript_dirs = ["/export/data/LDC2016T17","/export/data/LDC2017T04"]` must
be provided to the `prepare_gale_arabic` method.

This data is not available for free - your institution needs to have an LDC subscription.
"""

import logging
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional, Union

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.qa import trim_supervisions_to_recordings
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, check_and_rglob, is_module_available

# Test recordings from Kaldi split
# https://github.com/kaldi-asr/kaldi/blob/master/egs/gale_arabic/s5d/local/test/test_p2
TEST = [
    "ALAM_WITHEVENT_ARB_20070116_205800",
    "ALAM_WITHEVENT_ARB_20070206_205801",
    "ALAM_WITHEVENT_ARB_20070213_205800",
    "ALAM_WITHEVENT_ARB_20070227_205800",
    "ALAM_WITHEVENT_ARB_20070306_205800",
    "ALAM_WITHEVENT_ARB_20070313_205800",
    "ARABIYA_FROMIRAQ_ARB_20070216_175800",
    "ARABIYA_FROMIRAQ_ARB_20070223_175801",
    "ARABIYA_FROMIRAQ_ARB_20070302_175801",
    "ARABIYA_FROMIRAQ_ARB_20070309_175800",
]


def check_dependencies():
    if not is_module_available("pandas"):
        raise ImportError(
            "Gale Arabic data preparation requires the 'pandas' package to be installed. "
            "Please install it with 'pip install pandas' and try again"
        )


def prepare_gale_arabic(
    audio_dirs: List[Pathlike],
    transcript_dirs: List[Pathlike],
    output_dir: Optional[Pathlike] = None,
    absolute_paths: bool = True,
) -> Dict[str, Union[RecordingSet, SupervisionSet]]:
    """
    Prepare manifests for GALE Arabic Broadcast speech corpus.

    :param audio_dirs: List of paths to audio corpora.
    :param transcripts_dirs: List of paths to transcript corpora.
    :param output_dir: Directory where the manifests should be written. Can be omitted to avoid writing.
    :return: A dict with manifests. The keys are: ``{'recordings', 'supervisions'}``.
    """
    assert len(audio_dirs) == len(
        transcript_dirs
    ), "Paths to the same speech and transcript corpora must be provided"

    logging.info("Reading audio and transcript paths from provided dirs")
    # Some of the audio is wav while others are flac. Also, some recordings
    # may be repeated across corpora so we make a dict to avoid adding them
    # twice.
    audio_paths = defaultdict(
        Path,
        {
            p.stem: p
            for p in chain.from_iterable(
                [
                    check_and_rglob(dir, ext, strict=False)
                    for dir in audio_dirs
                    for ext in ["*.wav", "*.flac"]
                ]
            )
        },
    )
    transcript_paths = chain.from_iterable(
        [check_and_rglob(dir, "*.tdf") for dir in transcript_dirs]
    )
    transcript_paths = [p for p in transcript_paths]

    logging.info("Preparing recordings manifest")

    recordings = RecordingSet.from_recordings(
        Recording.from_file(p, relative_path_depth=None if absolute_paths else 3)
        for p in audio_paths.values()
    )

    logging.info("Preparing supervisions manifest")
    supervisions = SupervisionSet.from_segments(parse_transcripts(transcript_paths))

    # Some supervisions exceed recording boundaries, so here we trim them
    supervisions = trim_supervisions_to_recordings(recordings, supervisions)
    validate_recordings_and_supervisions(recordings, supervisions)

    manifests = defaultdict(dict)
    manifests["test"] = {
        "recordings": recordings.filter(lambda r: r.id in TEST),
        "supervisions": supervisions.filter(lambda s: s.recording_id in TEST),
    }
    manifests["train"] = {
        "recordings": recordings.filter(lambda r: r.id not in TEST),
        "supervisions": supervisions.filter(lambda s: s.recording_id not in TEST),
    }

    if output_dir is not None:
        logging.info("Writing manifests to JSON files")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for part in ["train", "test"]:
            manifests[part]["recordings"].to_json(
                output_dir / f"recordings_{part}.json"
            )
            manifests[part]["supervisions"].to_json(
                output_dir / f"supervisions_{part}.json"
            )

    return manifests


def parse_transcripts(transcript_paths: List[Path]) -> List[SupervisionSegment]:
    check_dependencies()
    import pandas as pd

    supervisions = []
    supervision_ids = set()
    for file in transcript_paths:
        df = pd.read_csv(
            file,
            delimiter="\t",
            skiprows=3,
            usecols=range(13),
            names=[
                "reco_id",
                "channel",
                "start",
                "end",
                "speaker",
                "gender",
                "dialect",
                "text",
                "section",
                "turn",
                "segment",
                "section_type",
                "su_type",
            ],
            dtype={
                "reco_id": str,
                "channel": int,
                "start": float,
                "end": float,
                "speaker": str,
                "text": str,
            },
            skipinitialspace=True,
            error_bad_lines=False,
            warn_bad_lines=True,
        )
        # Remove segments with no transcription
        df = df[df.speaker != "no speaker"]

        # some reco_id's end with .sph
        df["reco_id"] = df["reco_id"].apply(lambda x: x.strip().replace(".sph", ""))
        # some speaker names have `*` in them
        df["speaker"] = df["speaker"].apply(
            lambda x: x.replace("*", "").strip() if not pd.isnull(x) else x
        )
        df["text"] = df["text"].apply(lambda x: x.strip() if not pd.isnull(x) else x)
        for idx, row in df.iterrows():
            supervision_id = f"{row['reco_id']}-{row['speaker']}-{idx}"
            duration = round(row["end"] - row["start"], ndigits=8)
            if supervision_id in supervision_ids or duration <= 0:
                continue
            supervision_ids.add(supervision_id)
            supervisions.append(
                SupervisionSegment(
                    id=supervision_id,
                    recording_id=row["reco_id"],
                    start=row["start"],
                    duration=duration,
                    speaker=row["speaker"],
                    gender=row["gender"],
                    language="Arabic",
                    text=row["text"],
                    channel=row["channel"],
                    custom={
                        "dialect": row["dialect"],
                        "section": row["section"],
                        "turn": row["turn"],
                        "segment": row["segment"],
                        "section_type": row["section_type"],
                        "su_type": row["su_type"],
                    },
                )
            )
    return supervisions
