"""
Multi-Domain Cantonese Corpus (MDCC), consists of 73.6 hours of clean read speech paired with 
transcripts, collected from Cantonese audiobooks from Hong Kong. It comprises philosophy, 
politics, education, culture, lifestyle and family domains, covering a wide range of topics. 

Manuscript can be found at: https://arxiv.org/abs/2201.02419
"""

import logging
import zipfile
from pathlib import Path
from typing import Dict, Sequence, Union

from tqdm.auto import tqdm

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.qa import fix_manifests
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, is_module_available

MDCC_URL = "https://drive.google.com/file/d/1epfYMMhXdBKA6nxPgUugb2Uj4DllSxkn/view"

MDCC_PARTS = ["train", "valid", "test"]


def download_mdcc(target_dir: Pathlike, force_download: bool = False) -> Path:
    """
    Downloads the MDCC data from the Google Drive and extracts it.
    :param target_dir: the directory where MDCC data will be saved.
    :param force_download: if True, it will download the MDCC data even if it is already present.
    :return: the path to downloaded and extracted directory with data.
    """
    if not is_module_available("gdown"):
        raise ValueError("Please run 'pip install gdown' to download MDCC.")

    import gdown

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir = target_dir / "dataset"
    corpus_zip = corpus_dir.with_suffix(".zip")

    if not force_download and corpus_zip.exists():
        logging.info(f"{corpus_zip} already exists. Skipping download.")
    else:
        logging.info(f"Running: gdown --fuzzy {MDCC_URL}")
        gdown.download(MDCC_URL, str(corpus_zip), fuzzy=True, quiet=False)

    # Extract the zipped file
    if not corpus_dir.exists() or force_download:
        logging.info(f"Extracting {corpus_zip} to {target_dir}")
        with zipfile.ZipFile(corpus_zip) as zf:
            zf.extractall(path=target_dir)

    return corpus_dir


def prepare_mdcc(
    corpus_dir: Pathlike,
    dataset_parts: Union[str, Sequence[str]] = "all",
    output_dir: Pathlike = None,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Create RecordingSet and SupervisionSet manifests for MDCC from a raw corpus distribution.

    :param corpus_dir: Pathlike, the path to the extracted corpus.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'audio' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    audio_dir = corpus_dir / "audio"
    assert (audio_dir).is_dir(), f"Missing {audio_dir} in {corpus_dir}."
    manifests = {}

    if dataset_parts == "all" or dataset_parts[0] == "all":
        dataset_parts = MDCC_PARTS
    elif isinstance(dataset_parts, str):
        assert dataset_parts in MDCC_PARTS, f"Unknown dataset part: {dataset_parts}"
        dataset_parts = [dataset_parts]

    for part in dataset_parts:
        recordings = []
        supervisions = []

        metadata = corpus_dir / f"cnt_asr_{part}_metadata.csv"
        assert (metadata).is_file(), f"Missing {part} metadata in {corpus_dir}."

        # read cvs file in an ugly way as there are no more than 80k lines
        # and i don't want to depend on pandas
        with open(metadata, "r") as f:
            lines = f.readlines()

            # remove the header
            lines = lines[1:]

        for line in tqdm(lines, desc=f"Processing {part} metadata"):
            # audio_path, text_path, gender, duration
            audio_path, text_path, gender, _ = line.strip().split(",")
            audio_path = audio_dir / Path(audio_path).name
            text_path = corpus_dir / text_path

            recording_id = make_recording_id(Path(audio_path))
            recording = Recording.from_file(audio_path, recording_id=recording_id)
            recordings.append(recording)

            supervision_segment = SupervisionSegment(
                id=recording_id,
                recording_id=recording_id,
                start=0.0,
                duration=recording.duration,
                channel=0,
                text=text_path.read_text().strip(),
                gender=gender,
                language="yue",
            )
            supervisions.append(supervision_segment)

        recordings = RecordingSet.from_recordings(recordings)
        supervisions = SupervisionSet.from_segments(supervisions)
        recordings, supervisions = fix_manifests(recordings, supervisions)
        validate_recordings_and_supervisions(
            recordings=recordings, supervisions=supervisions
        )

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
            recordings.to_file(output_dir / f"mdcc_recordings_{part}.jsonl.gz")
            supervisions.to_file(output_dir / f"mdcc_supervisions_{part}.jsonl.gz")

        manifests[part] = {"recordings": recordings, "supervisions": supervisions}

    return manifests


def make_recording_id(path: Path) -> str:
    return f"mdcc_{path.stem}"
