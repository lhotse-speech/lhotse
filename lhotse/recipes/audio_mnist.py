"""
About the AudioMNIST corpus

The AudioMNIST dataset consists of 30000 audio recordings
(ca. 9.5 hours) of spoken digits (0-9) in English with 50 repetitions
per digit for each of the 60 different speakers. Recordings were
collected in quiet offices with a RØDE NT-USB microphone as
mono channel signal at a sampling frequency of 48kHz and were
saved in 16 bit integer format. In addition to audio recordings, meta
information including age (range: 22 - 61 years), gender (12 female
/ 48 male), origin and accent of all speakers were collected as well.
All speakers were informed about the intent of the data collection
and have given written declarations of consent for their participation prior
to their recording session.

We don't provide download_audio_mnist().

The data is publicly available at the following github repo:

    https://github.com/soerenab/AudioMNIST
"""

import logging
from itertools import groupby, repeat
from pathlib import Path
from typing import Dict, List, Tuple, Union

from tqdm.auto import tqdm

from lhotse import fix_manifests, validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.parallel import parallel_map
from lhotse.serialization import load_json, load_yaml
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, Seconds


def prepare_audio_mnist(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """Prepare manifests for the AudioMNIST corpus.

    :param: corpus_dir: We assume it is the github repo directory and it contains
      the following directories: data/{01,02,03,...,60}
    :param: output_dir: Directory where the manifests should be written.
    """
    in_data_dir = Path(corpus_dir) / "data"
    assert (Path(in_data_dir) / "audioMNIST_meta.txt").is_file()

    metadata = load_json(in_data_dir / "audioMNIST_meta.txt")
    assert len(metadata) == 60, len(metadata)
    for i in range(1, 61):
        assert f"{i:02}" in metadata, i

    recordings = RecordingSet.from_recordings(
        Recording.from_file(
            path=path,
            # converts:
            #   path/to/AudioMNIST/data/01/3_01_24.wav
            # to:
            #   3_01_24
            recording_id=path.stem,
        )
        for i in range(1, 61)
        for path in (in_data_dir / f"{i:02}").rglob("*.wav")
    )

    id2text = {
        "0": "zero",
        "1": "one",
        "2": "two",
        "3": "three",
        "4": "four",
        "5": "five",
        "6": "six",
        "7": "seven",
        "8": "eight",
        "9": "nine",
    }

    supervisions = []
    for r in recordings:
        digit, speaker_id, _ = r.id.split("_")

        supervisions.append(
            SupervisionSegment(
                id=r.id,
                recording_id=r.id,
                start=0,
                duration=r.duration,
                channel=0,
                text=id2text[digit],
                language="English",
                speaker=speaker_id,
                custom=metadata[speaker_id],
            )
        )

    supervisions = SupervisionSet.from_segments(supervisions)
    validate_recordings_and_supervisions(recordings, supervisions)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        supervisions.to_file(output_dir / "audio_mnist_supervisions.jsonl.gz")
        recordings.to_file(output_dir / "audio_mnist_recordings.jsonl.gz")

    return {"recordings": recordings, "supervisions": supervisions}
