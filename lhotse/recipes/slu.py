import glob
import json
import logging
from collections import defaultdict
from itertools import groupby
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

from tqdm import tqdm

from lhotse import fix_manifests, validate_recordings_and_supervisions
from lhotse.audio import AudioSource, Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, Seconds, is_module_available


def prepare_slu(
    corpus_dir: Pathlike, output_dir: Optional[Pathlike] = None
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:

    import pandas

    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "train": pandas.read_csv(
            str(corpus_dir) + "/data/train_data.csv", index_col=0, header=0
        ),
        "valid": pandas.read_csv(
            str(corpus_dir) + "/data/valid_data.csv", index_col=0, header=0
        ),
        "test": pandas.read_csv(
            str(corpus_dir) + "/data/test_data.csv", index_col=0, header=0
        ),
    }
    train_wavs = [
        str(corpus_dir) + "/" + path_to_wav
        for path_to_wav in data["train"]["path"].tolist()
    ]
    valid_wavs = [
        str(corpus_dir) + "/" + path_to_wav
        for path_to_wav in data["valid"]["path"].tolist()
    ]
    test_wavs = [
        str(corpus_dir) + "/" + path_to_wav
        for path_to_wav in data["test"]["path"].tolist()
    ]

    transcripts = {
        "train": data["train"]["transcription"].tolist(),
        "valid": data["valid"]["transcription"].tolist(),
        "test": data["test"]["transcription"].tolist(),
    }

    frames = {
        "train": list(
            i
            for i in zip(
                data["train"]["action"].tolist(),
                data["train"]["object"].tolist(),
                data["train"]["location"].tolist(),
            )
        ),
        "valid": list(
            i
            for i in zip(
                data["valid"]["action"].tolist(),
                data["valid"]["object"].tolist(),
                data["valid"]["location"].tolist(),
            )
        ),
        "test": list(
            i
            for i in zip(
                data["test"]["action"].tolist(),
                data["test"]["object"].tolist(),
                data["test"]["location"].tolist(),
            )
        ),
    }

    manifests = defaultdict(dict)
    for name, dataset in zip(
        ["train", "valid", "test"], [train_wavs, valid_wavs, test_wavs]
    ):
        recordings = []
        for wav in tqdm(dataset):
            recording = Recording.from_file(wav)
            recordings.append(recording)
        recording_set = RecordingSet.from_recordings(recordings)

        supervisions = []
        for id, recording in tqdm(enumerate(recording_set)):
            supervisions.append(
                SupervisionSegment(
                    id=id,
                    recording_id=recording.id,
                    start=0,
                    duration=recording.duration,
                    channel=0,
                    text=transcripts[name][id],
                    custom={"frames": frames[name][id]},
                )
            )
        supervision_set = SupervisionSet.from_segments(supervisions)
        recording_set, supervision_set = fix_manifests(recording_set, supervision_set)
        validate_recordings_and_supervisions(recording_set, supervision_set)
        manifests[name] = {
            "recordings": recording_set,
            "supervisions": supervision_set,
        }

    if output_dir is not None:
        for name in ["train", "valid", "test"]:
            manifests[name]["recordings"].to_file(
                output_dir / ("slu_recordings_" + name + ".jsonl.gz")
            )
            manifests[name]["supervisions"].to_file(
                output_dir / ("slu_supervisions_" + name + ".jsonl.gz")
            )
