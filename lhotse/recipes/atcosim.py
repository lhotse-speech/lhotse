"""
The ATCOSIM Air Traffic Control Simulation Speech corpus is a speech database of air traffic control (ATC) operator speech, provided by Graz University of Technology (TUG) and Eurocontrol Experimental Centre (EEC). It consists of ten hours of speech data, which were recorded during ATC real-time simulations using a close-talk headset microphone. The utterances are in English language and pronounced by ten non-native speakers. The database includes orthographic transcriptions and additional information on speakers and recording sessions. It was recorded an annotated by Konrad Hofbauer.

See https://www.spsc.tugraz.at/databases-and-tools/atcosim-air-traffic-control-simulation-speech-corpus.html for more details.
"""

import collections
import csv
import hashlib
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Dict, Optional, Union

from tqdm.auto import tqdm

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.qa import fix_manifests
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import (
    Pathlike,
    Seconds,
    compute_num_samples,
    is_module_available,
    resumable_download,
)


# note: https://www2.spsc.tugraz.at/ does not support Range request header (2023-05-10)
def download_atcosim(
    target_dir: Pathlike = ".", force_download: Optional[bool] = False
) -> Path:
    if not is_module_available("pycdlib"):
        raise ImportError("Please 'pip install pycdlib' first.")
    import pycdlib

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = "atcosim"
    iso_path = target_dir / f"{dataset_name}.iso"
    corpus_dir = target_dir / dataset_name
    completed_detector = corpus_dir / ".completed"
    if completed_detector.is_file():
        logging.info(f"Skipping {dataset_name} because {completed_detector} exists.")
        return corpus_dir
    resumable_download(
        f"https://www2.spsc.tugraz.at/databases/ATCOSIM/.ISO/{dataset_name}.iso",
        filename=iso_path,
        completed_file_size=2597789696,
        force_download=force_download,
    )
    if (
        hashlib.md5(open(iso_path, "rb").read()).hexdigest()
        != "cd5f0c82be46242a75d3382e340f6dca"
    ):
        raise RuntimeError("MD5 checksum does not match")
    shutil.rmtree(corpus_dir, ignore_errors=True)
    iso = pycdlib.PyCdlib()
    iso.open(iso_path)
    path_arg = (
        "udf"
        if iso.has_udf()
        else (
            "rr" if iso.has_rock_ridge() else ("joliet" if iso.has_joliet() else "iso")
        )
    ) + "_path"
    records = collections.deque([iso.get_record(**{path_arg: "/"})])
    while records:
        r = records.popleft()
        abs_path = iso.full_path_from_dirrecord(r, rockridge=path_arg == "rr_path")
        rel_path = abs_path.lstrip("/")
        if r.is_dir():
            os.makedirs(corpus_dir / rel_path)
            for child in iso.list_children(**{path_arg: abs_path}):
                if child and not (child.is_dot()) and not (child.is_dotdot()):
                    records.append(child)
        elif r.is_symlink():
            logging.warning("symlink not implemented")
        else:
            iso.get_file_from_iso(corpus_dir / rel_path, **{path_arg: abs_path})
    iso.close()
    completed_detector.touch()
    return corpus_dir


FIX_TYPOS = {
    "hm": "hmm",
    "ohh": "oh",
    "hallo": "hello",
    "viscinity": "vicinity",
}

FOREIGN_PATTERN = re.compile(r"<FL>\s*</FL>")
OFF_TALK_PATTERN = re.compile(r"<OT>(.*?)</OT>")
INTERRUPTED_PATTERN1 = re.compile(r"=(\w+)")
INTERRUPTED_PATTERN2 = re.compile(r"(\w+)=")
WHITESPACE_PATTERN = re.compile(r"  +")


def text_normalize(
    text: str,
    silence_sym: str,
    breath_sym: str,
    foreign_sym: str,
    partial_sym: str,  #  When None, will output partial words
    unknown_sym: str,
):

    text = OFF_TALK_PATTERN.sub(r"\1", text)

    result = []
    for w in text.split():
        if w[0] == "@" or w[0] == "~":
            result.append(w[1:])
        elif w in FIX_TYPOS:
            result.append(FIX_TYPOS[w])
        else:
            result.append(w)
    text = " ".join(result).upper()

    text = text.replace("[EMPTY]", silence_sym)
    text = text.replace("[HNOISE]", breath_sym)
    text = FOREIGN_PATTERN.sub(foreign_sym, text)

    if partial_sym == None:
        text = text.replace("=", "")
    else:
        text = INTERRUPTED_PATTERN1.sub(partial_sym, text)
        text = INTERRUPTED_PATTERN2.sub(partial_sym, text)

    for unk in ("[FRAGMENT]", "[NONSENSE]", "[UNKNOWN]"):
        text = text.replace(unk, unknown_sym)

    text = text.replace("AIR SPACE", "AIRSPACE")

    text = WHITESPACE_PATTERN.sub(" ", text)
    text = text.strip()

    return text


def fix_duration(duration: Seconds, sampling_rate: int) -> Seconds:
    """
    A handful of supervision durations do not compute to a round number of
    samples at the original recording sampling rate.

    This causes problem later using compute_num_frames(). Full description:
    https://github.com/lhotse-speech/lhotse/issues/1064

    Return: duration that computes to a round number of samples.
    """
    return compute_num_samples(duration, sampling_rate) / sampling_rate


def prepare_atcosim(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    silence_sym: Optional[str] = "",
    breath_sym: Optional[str] = "",
    foreign_sym: Optional[str] = "<unk>",
    partial_sym: Optional[str] = "<unk>",
    unknown_sym: Optional[str] = "<unk>",
) -> Dict[str, Union[RecordingSet, SupervisionSet]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions

    :param corpus_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param silence_sym: str, silence symbol
    :param breath_sym: str, breath symbol
    :param foreign_sym: str, foreign symbol.
    :param partial_sym: str, partial symbol. When set to None, will output partial words
    :param unknown_sym: str, unknown symbol
    :return: The RecordingSet and SupervisionSet with the keys 'audio' and 'supervisions'.
    """
    if not is_module_available("pandas"):
        raise ImportError("Please 'pip install pandas' first.")
    import pandas as pd

    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = corpus_dir / "TXTdata/fulldata.csv"
    df = pd.read_csv(csv_path, quoting=csv.QUOTE_NONE)

    with RecordingSet.open_writer(
        output_dir / f"atcosim_recordings_all.jsonl.gz",
        overwrite=True,
    ) as recs_writer, SupervisionSet.open_writer(
        output_dir / f"atcosim_supervisions_all.jsonl.gz",
        overwrite=True,
    ) as sups_writer:
        for idx, row in tqdm(
            df.iterrows(),
            desc="Preparing",
            total=len(df),
        ):
            if row.recording_corrupt:
                continue

            text = text_normalize(
                row.transcription,
                silence_sym=silence_sym,
                breath_sym=breath_sym,
                foreign_sym=foreign_sym,
                partial_sym=partial_sym,
                unknown_sym=unknown_sym,
            )

            if text == "":
                continue

            wav_path = (
                str(
                    corpus_dir
                    / "WAVdata"
                    / row.directory
                    / row.subdirectory
                    / row.filename
                )
                + ".wav"
            )
            recording = Recording.from_file(wav_path, recording_id=row.recording_id)
            length100 = int(row.length_sec * 100)
            segment = SupervisionSegment(
                id=f"atcosim_{row.filename}_{0:06d}_{length100:06d}",
                recording_id=row.recording_id,
                start=0.0,
                duration=fix_duration(row.length_sec, recording.sampling_rate),
                channel=0,
                language="English",
                text=text,
                speaker=row.speaker_id,
                gender=row.speaker_id[1].upper(),
                custom={"orig_text": row.transcription},
            )

            recs_writer.write(recording)
            sups_writer.write(segment)

    recordings = RecordingSet.from_jsonl_lazy(recs_writer.path)
    supervisions = SupervisionSet.from_jsonl_lazy(sups_writer.path)

    logging.warning(
        "Manifests are lazily materialized. You may want to call `lhotse.qa.fix_manifests()`"
        " to ensure that all supervisions fall within the corresponding recordings."
    )
    return recordings, supervisions
