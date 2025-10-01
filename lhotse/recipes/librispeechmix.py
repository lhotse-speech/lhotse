"""
LibriSpeechMix dataset preparation for Lhotse.

This recipe processes LibriSpeechMix metadata to create multi-speaker mixtures
from LibriSpeech recordings. LibriSpeechMix provides predefined
speaker combinations and timing information for creating consistent mixtures.

For more details, see:
- GitHub repository: https://github.com/NaoyukiKanda/LibriSpeechMix/
"""
import glob
import json
import logging
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import tqdm

from lhotse.cut.set import CutSet, MonoCut, mix
from lhotse.recipes.utils import manifests_exist, read_manifests_if_cached
from lhotse.utils import Pathlike

RATE = 16000


def _fetch_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)


def _fetch_bytes(url):
    req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
    with urllib.request.urlopen(req) as resp:
        return resp.read()


def download_github_dir(user, repo, path, branch="main", save_dir="."):
    api_url = f"https://api.github.com/repos/{user}/{repo}/contents/{path}?ref={branch}"
    files = _fetch_json(api_url)

    os.makedirs(save_dir, exist_ok=True)

    for file in files:
        file_path = os.path.join(save_dir, file["name"])
        if file["type"] == "file":
            with open(file_path, "wb") as f:
                f.write(_fetch_bytes(file["download_url"]))
        elif file["type"] == "dir":
            download_github_dir(user, repo, file["path"], branch, file_path)


def download_librispeechmix(
    target_dir: Pathlike = ".",
) -> Path:
    """Download LibriSpeechMix metadata."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = target_dir / "list"
    completed_detector = metadata_dir / ".completed"

    if completed_detector.is_file():
        logging.info(f"Skipping download because {completed_detector} exists.")
        return metadata_dir

    logging.info(
        f"Downloading https://github.com/NaoyukiKanda/LibriSpeechMix/tree/main/list to {metadata_dir}..."
    )
    download_github_dir("NaoyukiKanda", "LibriSpeechMix", "list", "main", metadata_dir)
    completed_detector.touch()
    return metadata_dir


def prepare_librispeechmix(
    librispeech_root_path: Pathlike,
    librispeechmix_metadata_path: Pathlike,
    output_dir: Optional[Pathlike] = None,
    num_jobs: Optional[int] = 1,
) -> Dict[str, Dict[str, CutSet]]:
    """
    Prepare LibriSpeechMix manifests for multi-speaker mixtures.

    Args:
        librispeech_root_path: Path to LibriSpeech manifests
        librispeechmix_metadata_path: Path to LibriSpeechMix metadata JSONL files
        output_dir: Directory to save manifests
        num_jobs: Number of parallel threads used for processing (default: 1)

    Returns:
        Dict with keys for each split containing CutSets
    """
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    manifests = {}

    # Find all JSONL metadata files
    metadata_files = glob.glob(f"{librispeechmix_metadata_path}/*.jsonl")

    # Collect all dataset parts
    dataset_parts = [Path(f).stem for f in metadata_files]

    # Maybe the manifests already exist
    if output_dir is not None:
        manifests = read_manifests_if_cached(
            dataset_parts=dataset_parts,
            output_dir=output_dir,
            prefix="librispeechmix",
            types=("cutset",),
        )

    for metadata_file in metadata_files:
        part_name = Path(metadata_file).stem

        # Check if manifests already exist and are cached
        if manifests_exist(
            part=part_name,
            output_dir=output_dir,
            prefix="librispeechmix",
            types=("cutset",),
        ):
            logging.info(
                f"LibriSpeechMix subset: {part_name} already prepared - skipping."
            )
            continue

        logging.info(f"Processing {part_name}...")

        librispeech_cutset_path = (
            Path(librispeech_root_path)
            / f"librispeech_cutset_{'-'.join(part_name.split('-')[:-1])}.jsonl.gz"
        )

        librispeech_cutset = CutSet.from_file(librispeech_cutset_path)

        def use_recording_id(cut):
            cut.id = cut.recording_id
            return cut

        librispeech_cutset = librispeech_cutset.map(use_recording_id).to_eager()
        # Process metadata file
        cuts = _process_librispeechmix_metadata(
            metadata_file, librispeech_cutset, num_jobs
        )

        cutset = CutSet.from_cuts(cuts)

        if output_dir is not None:
            cutset.to_file(output_dir / f"librispeechmix_cutset_{part_name}.jsonl.gz")

        manifests[part_name] = {
            "cutset": cutset,
        }

    return manifests


def _process_librispeechmix_metadata(
    metadata_file: str, librispeech_cutset: CutSet, num_jobs: int
) -> List[MonoCut]:
    """Process a LibriSpeechMix metadata JSONL file."""

    # Read JSONL file
    metadata_entries = []
    with open(metadata_file, "r") as f:
        for line in f:
            metadata_entries.append(json.loads(line.strip()))

    cuts = []

    logging.info(f"Processing {len(metadata_entries)} entries from {metadata_file}...")

    with ThreadPoolExecutor(max_workers=num_jobs) as ex:
        futures = [
            ex.submit(_process_metadata_entry, entry, librispeech_cutset)
            for entry in metadata_entries
        ]
        for f in tqdm.tqdm(as_completed(futures), total=len(futures)):
            result = f.result()
            if result is not None:
                cuts.append(result)

    return cuts


def _process_metadata_entry(
    entry: dict, librispeech_cutset: CutSet
) -> Optional[MonoCut]:
    """Process a single metadata entry from LibriSpeechMix."""

    # Extract information from metadata
    mixture_id = entry["id"].split("/")[-1]
    wavs = entry["wavs"]
    delays = entry["delays"]

    # Find corresponding cuts in LibriSpeech
    source_cuts = []
    for i, wav_path in enumerate(wavs):
        # Convert wav path to cut ID (remove extension and convert path separators)
        cut_id = Path(wav_path).stem

        cut = librispeech_cutset[cut_id]

        # Apply delay if specified
        delay = delays[i] if i < len(delays) else 0.0
        if delay > 0:
            cut = cut.pad(delay + cut.duration, direction="left")

        source_cuts.append(cut)

    if len(source_cuts) != len(wavs):
        raise ValueError("Not all mono cuts collected")

    # Create mixture by mixing all source cuts
    mixed_cut = source_cuts[0]
    for source_cut in source_cuts[1:]:
        mixed_cut = mix(mixed_cut, source_cut, preserve_id="left")

    # Update mixture ID
    mixed_cut.id = mixture_id

    return mixed_cut
