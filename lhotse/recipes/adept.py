"""
ADEPT: A Dataset for Evaluating Prosody Transfer
Torresquintero, Alexandra; Teh, Tian Huey; Wallis, Christopher G. R.; Staib, Marlene; Mohan, Devang S Ram;
Hu, Vivian; Foglianti, Lorenzo; Gao, Jiameng; King, Simon

The ADEPT dataset consists of prosodically-varied natural speech samples for evaluating prosody transfer in
english text-to-speech models. The samples include global variations reflecting emotion and interpersonal
attitude, and local variations reflecting topical emphasis, propositional attitude, syntactic phrasing and
marked tonicity.

Txt and wav files are organised according to the folder structure
{speech_class}/{subcategory_or_interpretation}/{filename}, where filename follows the naming convention
{speaker}_{utterance_id}. Speakers comprise 'ad00' (female voice) and 'ad01' (male voice). For classes
with multiple interpretations, we provide the interpretations used in the disambiguation tasks in
'adept_prompts.json'.

The corpus only includes prosodic variations that listeners are able to distinguish with reasonable accuracy,
and we report these figures as a benchmark against which text-to-speech prosody transfer can be compared.
More details can be found in pre-print about the dataset (https://arxiv.org/abs/2106.08321).

Source: https://zenodo.org/record/5117102#.YVsHlS-B3T9
"""
import json
import logging
import shutil
import zipfile
from pathlib import Path
from typing import Optional

from lhotse import (
    Recording,
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
    validate_recordings_and_supervisions,
)
from lhotse.utils import Pathlike, urlretrieve_progress

ADEPT_URL = "https://zenodo.org/record/5117102/files/ADEPT.zip"


def download_adept(
    target_dir: Pathlike = ".",
    force_download: bool = False,
) -> Path:
    """
    Download and untar the ADEPT dataset.

    :param target_dir: Pathlike, the path of the dir to storage the dataset.
    :param force_download: Bool, if True, download the tars no matter if the tars exist.
    :return: the path to downloaded and extracted directory with data.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    corpus_dir = target_dir / "ADEPT"
    completed_detector = corpus_dir / ".completed"
    if completed_detector.is_file():
        logging.info(f"Skipping downloading ADEPT because {completed_detector} exists.")
        return corpus_dir
    # Maybe-download the archive.
    zip_name = "ADEPT.zip"
    zip_path = target_dir / zip_name
    if force_download or not zip_path.is_file():
        urlretrieve_progress(
            ADEPT_URL, filename=zip_path, desc=f"Downloading {zip_name}"
        )
    # Remove partial unpacked files, if any, and unpack everything.
    shutil.rmtree(corpus_dir, ignore_errors=True)
    with zipfile.ZipFile(zip_path) as zip_f:
        zip_f.extractall(path=corpus_dir)
    completed_detector.touch()

    return corpus_dir


def prepare_adept(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
):
    """
    Returns the manifests which consist of the Recordings and Supervisions.
    When all the manifests are available in the ``output_dir``, it will simply read and return them.

    :param corpus_dir: Pathlike, the path of the data dir.
    :param dataset_parts: string or sequence of strings representing dataset part names,
        e.g. 'train-clean-100', 'train-clean-5', 'dev-clean'.
        By default we will infer which parts are available in ``corpus_dir``.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'audio' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    recordings = RecordingSet.from_recordings(
        Recording.from_file(
            path=path,
            # converts:
            #   path/to/ADEPT/wav_44khz/propositional_attitude/surprise/ad01_0204.wav
            # to:
            #   propositional_attitude_surprise_ad01_0204
            recording_id=str(path.relative_to(path.parent.parent.parent))[:-4].replace(
                "/", "_"
            ),
        )
        for path in (corpus_dir / "wav_44khz").rglob("*.wav")
    )

    supervisions = []

    with open(corpus_dir / "adept_prompts.json") as f:
        interpretation_map = json.load(f)

    for path in (corpus_dir / "txt").rglob("*.txt"):
        annotation_type, label, prompt_id = str(
            path.relative_to(path.parent.parent.parent)
        )[:-4].split("/")
        speaker_id = "ADEPT_" + prompt_id.split("_")[0]
        recording_id = "_".join((annotation_type, label, prompt_id))
        interpretation_group = interpretation_map.get(annotation_type)
        interpretation = (
            interpretation_group[prompt_id][label] if interpretation_group else None
        )
        recording = recordings[recording_id]
        custom = {"type": annotation_type, "label": label, "prompt_id": prompt_id}
        if interpretation:
            # label is "interpretation_1", "interpretation_2", ..., "middle", "end", etc
            # Interpretations' labels meaning is defined by their textual realisation:
            #  {..., "middle": "Galleries are WHAT on Thursdays?", "end": "Galleries are free WHEN?"}
            custom["text"] = interpretation
        supervisions.append(
            SupervisionSegment(
                id=recording_id,
                recording_id=recording_id,
                start=0,
                duration=recording.duration,
                channel=0,
                text=path.read_text(),
                language="English",
                speaker=speaker_id,
                custom=custom,
            )
        )

    supervisions = SupervisionSet.from_segments(supervisions)
    validate_recordings_and_supervisions(recordings, supervisions)

    if output_dir is not None:
        output_dir = Path(output_dir)
        supervisions.to_file(output_dir / "adept_supervisions.json")
        recordings.to_file(output_dir / "adept_recordings.json")

    return {"recordings": recordings, "supervisions": supervisions}
