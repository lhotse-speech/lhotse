"""
otoSpeech Dataset Preparation Recipe for Lhotse

Dataset Details:
- URL: https://huggingface.co/datasets/otoearth/otoSpeech-full-duplex-processed-141h
- Content: Full-duplex, spontaneous multi-speaker conversations.
- Purpose: Designed for training and benchmarking S2S (speech-to-speech) or dialogue models.
- Splits: This dataset provides ONLY the `train` split.

Pseudo Labels:
- The `seglst.json` labels downloaded from Google Drive are pseudo labels generated
  using the Parakeet v3 model.
"""

import json
import logging
import os
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, Union

from tqdm import tqdm

from lhotse import (
    Recording,
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
    fix_manifests,
)
from lhotse.supervision import AlignmentItem
from lhotse.utils import Pathlike

# Set up the logger
logger = logging.getLogger(__name__)


def download_oto_speech(
    target_dir: Pathlike = ".",
    parts: Tuple[str, ...] = ("train",),
    version: str = "full-duplex-processed-141h",
    force_download: bool = False,
) -> Path:
    """
    Downloads the otoSpeech audio dataset from HuggingFace and pseudo labels from Google Drive.

    Args:
        target_dir: Path to the directory where the dataset will be stored.
        parts: Which splits to download (Note: only "train" is officially provided).
        version: The dataset version suffix.
        force_download: Whether to force re-download from HuggingFace and GDrive.

    Returns:
        The path to the target directory.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as import_error:
        raise RuntimeError("Install via: pip install huggingface_hub") from import_error

    try:
        import gdown
    except ImportError as e:
        raise RuntimeError("Install via: pip install gdown") from e

    hugging_face_token = os.getenv("HF_TOKEN")
    if not hugging_face_token:
        raise RuntimeError("HF_TOKEN environment variable not found.")

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download HuggingFace Dataset
    for part in parts:
        if part != "train":
            logger.warning(
                f"Dataset only provides a 'train' split. Downloading '{part}' may fail."
            )

        logger.info(f"Downloading dataset shard for: {part}")
        snapshot_download(
            repo_id=f"otoearth/otoSpeech-{version}",
            repo_type="dataset",
            local_dir=target_dir,
            force_download=force_download,
            allow_patterns=[f"data/{part}/*"],
            token=hugging_face_token,
        )

    # 2. Download Pseudo Labels from Google Drive
    labels_path = target_dir / "seglst.json"
    if not labels_path.exists() or force_download:
        logger.info(
            "Downloading Parakeet v3 pseudo labels (seglst.json) from Google Drive..."
        )
        url = "https://drive.google.com/file/d/16htmj5O14D51C-EjOUMF_cXOxo6vruui/view?usp=sharing"
        gdown.download(url, str(labels_path), quiet=False, fuzzy=True)
    else:
        logger.info(
            "Parakeet v3 pseudo labels (seglst.json) already exist. Skipping download."
        )

    return target_dir


def extract_and_flatten_tar(tar_path: Path, extract_dir: Path):
    """Extracts a tar file, flattens contents, and caches the result using a marker."""
    marker_file = extract_dir / f"{tar_path.name}.done"

    # Cache check: if the marker exists, we already unpacked this shard
    if marker_file.exists():
        return

    with tarfile.open(tar_path) as tar:
        tar.extractall(path=extract_dir)

    # Flatten structure and ignore marker files
    for p in extract_dir.rglob("*"):
        if p.is_file() and p.parent != extract_dir and p.suffix != ".done":
            target_path = extract_dir / p.name
            if not target_path.exists():
                p.rename(target_path)

    # Create the marker file to register this tar as "done"
    marker_file.touch()


def prepare_oto_speech(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    parts: Tuple[str, ...] = ("train",),
    target_sr: int = 16000,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """Prepares the dataset, utilizing Lhotse's lazy resampling and extraction caching."""
    corpus_dir = Path(corpus_dir)
    data_dir = corpus_dir / "data"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_path = corpus_dir / "seglst.json"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Labels not found at {labels_path}. Please run download_oto_speech() first."
        )

    with open(labels_path, "r", encoding="utf-8") as f:
        logger.info(f"Loading Parakeet v3 pseudo metadata from {labels_path}...")
        label_data = json.load(f)

    manifests = defaultdict(dict)

    for part in parts:
        if part != "train":
            logger.warning(
                f"Preparing split '{part}', but standard otoSpeech only guarantees 'train'."
            )

        part_dir = data_dir / part
        unpacked_dir = part_dir / "unpacked"
        unpacked_dir.mkdir(parents=True, exist_ok=True)

        # 1. Untar the downloaded shards (cached)
        logger.info(f"--- [1/3] Extracting {part} ---")
        tar_files = list(part_dir.glob("*.tar"))
        for tar_path in tqdm(tar_files, desc="Extracting tar files"):
            extract_and_flatten_tar(tar_path, unpacked_dir)

        # 2. Create RecordingSet and apply lazy resampling
        logger.info(
            f"--- [2/3] Building RecordingSet (with lazy {target_sr}Hz resampling) ---"
        )
        audio_paths = list(unpacked_dir.glob("*.flac"))

        recordings = RecordingSet.from_recordings(
            Recording.from_file(p) for p in tqdm(audio_paths, desc="Parsing audio")
        )
        recordings = recordings.resample(target_sr)

        # 3. Create SupervisionSet from the GDrive JSON
        logger.info("--- [3/3] Building SupervisionSet ---")
        supervisions = []

        for idx, seg in tqdm(
            enumerate(label_data), total=len(label_data), desc="Parsing labels"
        ):
            rec_id = seg["session_id"]

            if rec_id not in recordings:
                continue

            start = seg["start_time"]
            end = seg["end_time"]
            duration = round(end - start, 4)

            if duration <= 0:
                logger.warning(
                    f"Skipped segment for rec: {rec_id} at {start} due to 0 duration"
                )
                continue

            alignments = []
            if "word_alignment" in seg:
                for w_text, w_start, w_end in seg["word_alignment"]:
                    alignments.append(
                        AlignmentItem(
                            symbol=w_text,
                            start=round(w_start - start, 4),
                            duration=round(w_end - w_start, 4),
                        )
                    )

            supervisions.append(
                SupervisionSegment(
                    id=f"{rec_id}-{idx}",
                    recording_id=rec_id,
                    start=start,
                    duration=duration,
                    channel=0,
                    text=seg["words"],
                    speaker=seg["speaker"],
                    language="en",
                    alignment={"word": alignments} if alignments else None,
                )
            )

        supervision_set = SupervisionSet.from_segments(supervisions)

        logger.info("Fixing and validating manifests...")
        recordings, supervision_set = fix_manifests(recordings, supervision_set)

        recordings_path = output_dir / f"oto_recordings_{part}.jsonl.gz"
        supervisions_path = output_dir / f"oto_supervisions_{part}.jsonl.gz"

        recordings.to_file(recordings_path)
        supervision_set.to_file(supervisions_path)

        logger.info(f"Saved to:\n - {recordings_path}\n - {supervisions_path}")

        manifests[part] = {"recordings": recordings, "supervisions": supervision_set}

    return dict(manifests)
