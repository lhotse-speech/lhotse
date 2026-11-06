"""
LibriMix dataset preparation for Lhotse.

This recipe replicates the LibriMix dataset preparation by manipulating existing recordings
instead of generating and saving new audio files. LibriMix is an open source dataset for
source separation in noisy environments, derived from LibriSpeech signals (clean subset)
and WHAM noise.

The original dataset supports:
- Multiple sources (2 or 3 speakers) in mixtures
- Different sample rates (typically 16kHz and 8kHz)
- Different mixture modes:
  - min: mixture ends when the shortest source ends
  - max: mixture ends when the longest source ends
- Different mixture types:
  - mix_clean: utterances only
  - mix_both: utterances + noise
  - mix_single: 1 utterance + noise

**Current Limitations:**
This Lhotse recipe currently supports only:
- 16kHz sample rate
- 'max' mode (mixture ends when the longest source ends)

**Important Note on Quantization:**
The original LibriMix recipe introduces a quantization error when saving audio files via soundfile,
which by default uses PCM_16 format. If you need to replicate the exact quantization behavior from
the original recipe, you can apply the following transformation:

```python
import tempfile
import soundfile as sf

with tempfile.NamedTemporaryFile(suffix=".wav") as f:
    sf.write(f.name, cut.load_audio().T, 16000)
    audio_quantized, sr = sf.read(f.name)
```

Unlike the original LibriMix generation which creates ~430GB for Libri2Mix and ~332GB for Libri3Mix,
this recipe works with existing LibriSpeech and WHAM recordings and creates virtual mixtures,
making it much more storage efficient.

For more details about LibriMix, see:
- GitHub repository: https://github.com/JorisCos/LibriMix/
- Paper: "LibriMix: An Open-Source Dataset for Generalizable Speech Separation"
  https://arxiv.org/pdf/2005.11262.pdf

Citation:
@misc{cosentino2020librimix,
    title={LibriMix: An Open-Source Dataset for Generalizable Speech Separation},
    author={Joris Cosentino and Manuel Pariente and Samuele Cornell and Antoine Deleforge and Emmanuel Vincent},
    year={2020},
    eprint={2005.11262},
    archivePrefix={arXiv},
    primaryClass={eess.AS}
}
"""

import csv
import json
import logging
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tqdm

import lhotse
from lhotse.audio import AudioSource, Recording, RecordingSet
from lhotse.audio.backend import info, save_audio
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


def download_librimix(
    target_dir: Pathlike = ".",
) -> Path:
    """Download LibriMix metadata."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = target_dir / "metadata"
    completed_detector = metadata_dir / ".completed"

    if completed_detector.is_file():
        logging.info(f"Skipping download because {completed_detector} exists.")
        return metadata_dir

    logging.info(
        f"Downloading https://github.com/JorisCos/LibriMix/tree/master/metadata to {metadata_dir}..."
    )
    os.makedirs(metadata_dir, exist_ok=True)
    download_github_dir("JorisCos", "LibriMix", "metadata", "master", metadata_dir)
    completed_detector.touch()
    return metadata_dir


def prepare_librimix(
    librispeech_root_path: Pathlike,
    wham_recset_root_path: Pathlike,
    librimix_metadata_path: Pathlike,
    workdir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    n_src: int = 2,
    num_jobs: int = 1,
) -> Dict[str, Dict[str, CutSet]]:
    """
    Prepare LibriMix manifests for multi-speaker mixtures.

    :param librispeech_root_path: Path to the LibriSpeech manifests
        (``librispeech_cuts_<part>.jsonl.gz``, written by :func:`lhotse.recipes.librispeech.prepare_librispeech`).
    :param wham_recset_root_path: Path to the WHAM noise manifests (``wham_recordings_<tr|cv|tt>.jsonl.gz``).
    :param librimix_metadata_path: Path to the LibriMix metadata.
    :param output_dir: Optional directory to save the manifests in.
    :param workdir: Working directory for temporary files.
    :param n_src: Number of sources to mix.
    :param num_jobs: Number of parallel threads used for processing.
    :return: a dict whose keys are the parts ``train-100``, ``train-360``, ``dev``, ``test`` (mixtures of
        utterances only) and the same names with a ``-noisy`` suffix (utterances and noise).
        Each value is ``{"cuts": CutSet}``, since the mixtures cannot be decomposed into
        recordings and supervisions. With ``output_dir`` set, the cuts are written to
        ``librimix-<n_src>src_cuts_<part>.jsonl.gz`` and returned lazily.
    """
    logging.warning(
        "The original LibriMix recipe introduces a quantization error when saving audio files via soundfile, which by default uses PCM_16 format. If you need to replicate the exact quantization behavior from the original recipe, you can save and load the audio using a temporary file as shown in the docstring of this function."
    )
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    if workdir is not None:
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)

    prefix = f"librimix-{n_src}src"

    n_src_meta_root = Path(librimix_metadata_path) / f"Libri{n_src}Mix"
    md_filename_list = sorted(
        file
        for file in os.listdir(n_src_meta_root)
        if "info" not in file and file != ".completed"
    )
    # md_filename: libri2mix_train-100.csv -> part: train-100
    parts = {md: _part_name(md) for md in md_filename_list}
    dataset_parts = [
        name for part in parts.values() for name in (part, f"{part}-noisy")
    ]

    # Maybe the manifests already exist: we can read them and save a bit of preparation time.
    manifests = (
        read_manifests_if_cached(
            dataset_parts=dataset_parts,
            output_dir=output_dir,
            prefix=prefix,
            types=("cuts",),
        )
        or {}
    )

    # Load WHAM recordings with speed augmentation
    wham_recsets = _load_wham_recordings(wham_recset_root_path)

    for md_filename, part_name in parts.items():
        part_name_noisy = f"{part_name}-noisy"

        # Check if manifests already exist and are cached
        if all(
            manifests_exist(
                part=name, output_dir=output_dir, prefix=prefix, types=("cuts",)
            )
            for name in (part_name, part_name_noisy)
        ):
            logging.info(
                f"LibriMix subset: {part_name} and {part_name_noisy} already prepared - skipping."
            )
            continue

        clean_cuts, noisy_cuts = _process_metadata_file(
            md_filename,
            n_src_meta_root,
            n_src,
            librispeech_root_path,
            wham_recsets,
            workdir,
            num_jobs,
        )

        # As we need to keep MixedCuts together we cannot decompose to rec- and supsets.
        for name, cuts in ((part_name, clean_cuts), (part_name_noisy, noisy_cuts)):
            cutset = CutSet.from_cuts(cuts)
            if output_dir is not None:
                path = output_dir / f"{prefix}_cuts_{name}.jsonl.gz"
                cutset.to_file(path)
                cutset = CutSet.from_jsonl_lazy(path)
            manifests[name] = {"cuts": cutset}

    return manifests


def _part_name(md_filename: str) -> str:
    """``libri2mix_train-100.csv`` -> ``train-100``"""
    return Path(md_filename).stem.split("_", maxsplit=1)[1]


# LibriMix split -> LibriSpeech part (as named in ``librispeech_cuts_<part>.jsonl.gz``).
LIBRISPEECH_PARTS = {
    "train-100": "train-clean-100",
    "train-360": "train-clean-360",
    "dev": "dev-clean",
    "test": "test-clean",
}


def _load_wham_recordings(wham_recset_root_path: Pathlike) -> Dict[str, RecordingSet]:
    """Load WHAM recordings with speed augmentation for training set."""
    wham_splits = [
        ("train", "wham_recordings_tr.jsonl.gz"),
        ("dev", "wham_recordings_cv.jsonl.gz"),
        ("test", "wham_recordings_tt.jsonl.gz"),
    ]
    speed_factors = [0.8, 1.0, 1.2]

    wham_recsets = {
        key: lhotse.load_manifest(Path(wham_recset_root_path) / split)
        for key, split in wham_splits
    }
    wham_recsets["train"] = _augment_wham(wham_recsets["train"], speed_factors)
    return wham_recsets


def _augment_wham(recset: RecordingSet, speed_factors: List[float]) -> RecordingSet:
    """Apply speed augmentation to WHAM recordings."""
    import re

    def fix_rec_ids(recording: Recording) -> Recording:
        recording.id = re.sub(r"_sp(\d+)\.(\d+)$", r"sp\1\2", recording.id)
        return recording

    new_recset = []
    for speed_factor in speed_factors:
        if speed_factor != 1.0:
            augmented_recset = recset.perturb_speed(speed_factor)
        else:
            augmented_recset = recset
        augmented_recset = augmented_recset.map(fix_rec_ids)
        new_recset.extend(augmented_recset)
    return RecordingSet.from_recordings(new_recset)


def _extend_noise(noise: np.ndarray, max_length: int) -> np.ndarray:
    """Concatenate noise using Hanning window."""
    noise_ex = noise
    window = np.hanning(RATE + 1)
    i_w = window[: len(window) // 2 + 1]
    d_w = window[len(window) // 2 :: -1]

    while len(noise_ex) < max_length:
        noise_ex = np.concatenate(
            (
                noise_ex[: len(noise_ex) - len(d_w)],
                np.multiply(noise_ex[len(noise_ex) - len(d_w) :], d_w)
                + np.multiply(noise[: len(i_w)], i_w),
                noise[len(i_w) :],
            )
        )
    return noise_ex[:max_length]


def _process_row(
    row: dict,
    librispeech_cutset: CutSet,
    wham_recset: RecordingSet,
    n_src: int,
    workdir: Optional[Path],
):
    """Process a single row from metadata CSV."""
    # Extract source information
    srcs = []
    gains = []
    for src in range(1, n_src + 1):
        srcs.append(Path(row[f"source_{src}_path"]).stem)
        gains.append(row[f"source_{src}_gain"])

    srcs = librispeech_cutset.subset(cut_ids=srcs)
    normalized_cuts = [src.perturb_volume(gain) for src, gain in zip(srcs, gains)]

    # Build clean mix
    clean_mix = normalized_cuts.pop()
    while normalized_cuts:
        clean_mix = mix(clean_mix, normalized_cuts.pop())
    clean_mix.id = row["mixture_ID"]

    # Process noise
    noise_id = Path(row["noise_path"]).stem
    noise_rec = wham_recset[noise_id]
    noise_gain = row[f"noise_gain"]
    noise_rec_perturbed = noise_rec.perturb_volume(noise_gain)

    if noise_rec_perturbed.duration < clean_mix.duration:
        noise_rec_perturbed = _extend_noise_recording(
            noise_rec_perturbed, clean_mix, row["mixture_ID"], workdir
        )

    noise_cut = MonoCut(
        id="noise",
        start=0,
        duration=clean_mix.duration,
        channel=0,
        recording=noise_rec_perturbed,
    )
    noisy_mix = mix(clean_mix, noise_cut, preserve_id="left")

    return clean_mix, noisy_mix


def _extend_noise_recording(
    noise_rec: Recording, clean_mix: MonoCut, mixture_id: str, workdir: Optional[Path]
) -> Recording:
    """Extend noise recording to match clean mix duration."""
    if workdir is None:
        workdir = Path(".")

    save_to = workdir / f"{noise_rec.id}_{mixture_id}.wav"
    if not save_to.exists():
        noise_array = noise_rec.load_audio()
        if noise_array.ndim > 1:
            noise_array = noise_array[0]

        extended_noise = _extend_noise(
            noise_array, int(clean_mix.duration * clean_mix.sampling_rate)
        )
        save_audio(
            dest=save_to, src=extended_noise, sampling_rate=noise_rec.sampling_rate
        )

    noise_rec_info = info(save_to)
    new_source = AudioSource(type="file", channels=[0], source=str(save_to))

    return Recording(
        id=noise_rec.id,
        sources=[new_source],
        sampling_rate=noise_rec_info.samplerate,
        num_samples=noise_rec_info.frames,
        duration=noise_rec_info.duration,
    )


def _read_metadata_csv(csv_path: Path) -> List[dict]:
    """
    Read LibriMix metadata using Python's standard csv library and cast gain fields to float.
    """
    rows: List[dict] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Normalize/convert values as needed
            for k, v in list(row.items()):
                if k.endswith("_gain"):
                    row[k] = float(v)
            rows.append(row)
    return rows


def _process_metadata_file(
    md_filename: str,
    n_src_meta_root: Path,
    n_src: int,
    librispeech_root_path: Path,
    wham_recsets: Dict[str, RecordingSet],
    workdir: Optional[Path],
    num_jobs: int,
) -> Tuple[List[MonoCut], List[MonoCut]]:
    """Process a single metadata file and return clean and noisy cuts."""
    csv_path = n_src_meta_root / md_filename

    rows = _read_metadata_csv(csv_path)

    librispeech_cutset = lhotse.load_manifest(
        Path(librispeech_root_path)
        / f"librispeech_cuts_{LIBRISPEECH_PARTS[_part_name(md_filename)]}.jsonl.gz"
    )

    split_name = "".join(Path(md_filename).stem.split("_")[1:]).split("-")[0]
    wham_recset = wham_recsets[split_name]

    clean_cuts = []
    noisy_cuts = []

    logging.info(f"Processing {md_filename}...")
    with ThreadPoolExecutor(max_workers=num_jobs) as ex:
        futures = [
            ex.submit(
                _process_row, row, librispeech_cutset, wham_recset, n_src, workdir
            )
            for row in rows
        ]
        for f in tqdm.tqdm(futures):
            clean_mix, noisy_mix = f.result()
            clean_cuts.append(clean_mix)
            noisy_cuts.append(noisy_mix)

    return clean_cuts, noisy_cuts
