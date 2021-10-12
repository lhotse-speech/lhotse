"""
The GALE Mandarin Broadcast news corpus consists of the following LDC corpora:

Audio: LDC2013S08, LDC2013S04, LDC2014S09, LDC2015S06, LDC2015S13, LDC2016S03
Text: LDC2013T20, LDC2013T08, LDC2014T28, LDC2015T09, LDC2015T25, LDC2016T12

# Training:  Testing: 

The `S` corpora contain speech data and the `T` corpora contain the corresponding
transcriptions. This recipe prepares any subset of these corpora provided as
arguments, but pairs of speech and transcript corpora must be present. E.g.
to only prepare phase 3 news speech, the arguments 
`audio_dirs = ["/export/data/LDC2013S08","/export/data/LDC2014S09"]` and 
`transcript_dirs = ["/export/data/LDC2013T20","/export/data/LDC2014T28"]` must
be provided to the `prepare_gale_mandarin` method.

This data is not available for free - your institution needs to have an LDC subscription.
"""

import logging
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional, Union
from urllib.request import urlopen

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.qa import trim_supervisions_to_recordings
from lhotse.recipes.nsc import check_dependencies
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, check_and_rglob, is_module_available

# Dev recording ids will be downloaded from the Kaldi repo
KALDI_BASE_URL = (
    "https://github.com/kaldi-asr/kaldi/blob/master/egs/gale_mandarin/s5/local/test."
)
TEST_FILE_URLS = [
    KALDI_BASE_URL + name
    for name in [
        "LDC2013S04",
        "LDC2013S08",
        "LDC2014S09",
        "LDC2015S06",
        "LDC2015S13",
        "LDC2016S03",
    ]
]


def check_dependencies(segment_words: Optional[bool] = False):
    if not is_module_available("pandas"):
        raise ImportError(
            "GALE Mandarin data preparation requires the 'pandas' package to be installed. "
            "Please install it with 'pip install pandas' and try again."
        )

    if segment_words and not is_module_available("jieba"):
        raise ImportError(
            "The '--segment-words' option requires the 'jieba' package to be installed. "
            "Please install it with 'pip install jieba' and try again."
        )


def prepare_gale_mandarin(
    audio_dirs: List[Pathlike],
    transcript_dirs: List[Pathlike],
    output_dir: Optional[Pathlike] = None,
    absolute_paths: Optional[bool] = True,
    segment_words: Optional[bool] = False,
) -> Dict[str, Union[RecordingSet, SupervisionSet]]:
    """
    Prepare manifests for GALE Mandarin Broadcast speech corpus.

    :param audio_dirs: List of paths to audio corpora.
    :param transcripts_dirs: List of paths to transcript corpora.
    :param output_dir: Directory where the manifests should be written. Can be omitted to avoid writing.
    :param absolute_paths: Wheter to write absolute paths to audio sources (default = False)
    :param segment_words: Use `jieba` package to perform word segmentation (default = False)
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

    logging.info("Preparing recordings manifest")

    recordings = RecordingSet.from_recordings(
        Recording.from_file(p, relative_path_depth=None if absolute_paths else 3)
        for p in audio_paths.values()
    )

    logging.info("Preparing supervisions manifest")
    supervisions = SupervisionSet.from_segments(
        parse_transcripts(transcript_paths, segment_words=segment_words)
    ).filter(lambda s: s.recording_id in audio_paths)

    # Some supervisions exceed recording boundaries, so here we trim them
    supervisions = trim_supervisions_to_recordings(recordings, supervisions)
    validate_recordings_and_supervisions(recordings, supervisions)

    TEST = [
        line.decode("utf-8").strip() for url in TEST_FILE_URLS for line in urlopen(url)
    ]

    manifests = defaultdict(dict)
    manifests["dev"] = {
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
        for part in ["train", "dev"]:
            manifests[part]["recordings"].to_json(
                output_dir / f"recordings_{part}.json"
            )
            manifests[part]["supervisions"].to_json(
                output_dir / f"supervisions_{part}.json"
            )

    return manifests


def parse_transcripts(
    transcript_paths: List[Path], segment_words: Optional[bool] = False
) -> List[SupervisionSegment]:
    check_dependencies(segment_words)
    import pandas as pd

    if segment_words:
        import jieba
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
        # We only keep sections which have some transcriptions
        df = df[df.section_type != "nontrans"]

        # some reco_id's end with `.sph` or `(1)`
        df["reco_id"] = df["reco_id"].apply(
            lambda x: x.strip().replace("(1)", "").replace(".sph", "")
        )
        # some speaker names have `*` in them
        df["speaker"] = df["speaker"].apply(
            lambda x: x.replace("#", "").strip() if not pd.isnull(x) else x
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
                    language="Mandarin",
                    text=row["text"]
                    if not segment_words
                    else " ".join(jieba.cut(row["text"])),
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
