from pathlib import Path
from typing import Optional

import click

from lhotse import CutSet, Seconds
from lhotse.bin.modes.cli_base import cli
from lhotse.features import Fbank, FeatureExtractor, create_default_feature_extractor
from lhotse.features.base import FEATURE_EXTRACTORS
from lhotse.features.io import (
    available_storage_backends,
    default_features_storage_backend_name,
    get_writer,
)
from lhotse.utils import Pathlike


@cli.group()
def feat():
    """Feature extraction related commands."""
    import torch

    # Speeds up multiprocess feature extraction and avoids deadlocks.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


@feat.command(context_settings=dict(show_default=True))
@click.argument("output_config", type=click.Path())
@click.option(
    "-f",
    "--feature-type",
    type=click.Choice(list(FEATURE_EXTRACTORS)),
    default="fbank",
    help="Which feature extractor type to use.",
)
def write_default_config(output_config: Pathlike, feature_type: str):
    """Save a default feature extraction config to OUTPUT_CONFIG."""
    create_default_feature_extractor(feature_type).to_yaml(output_config)


@feat.command(context_settings=dict(show_default=True))
@click.argument("cutset", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_cutset", type=click.Path())
@click.argument("storage_path", type=click.Path())
@click.option(
    "-f",
    "--feature-manifest",
    type=click.Path(exists=True, dir_okay=False),
    help="Optional manifest specifying feature extractor configuration.",
)
@click.option(
    "--storage-type",
    type=click.Choice(available_storage_backends()),
    default=default_features_storage_backend_name(),
    help="Select a storage backend for the feature matrices. "
    "The default is numpy_files; lilcom_chunky is preferred for smaller "
    "on-disk storage when lilcom is installed. See also: "
    "lhotse list-storage-backends.",
)
@click.option(
    "-j", "--num-jobs", type=int, default=1, help="Number of parallel processes."
)
def extract_cuts(
    cutset: Pathlike,
    output_cutset: Pathlike,
    storage_path: Pathlike,
    feature_manifest: Optional[Pathlike],
    storage_type: str,
    num_jobs: int,
):
    """
    Extract features for cuts in a given CUTSET manifest.
    The features are stored in STORAGE_PATH, and the output manifest
    with features is stored in OUTPUT_CUTSET.
    """
    cuts = CutSet.from_file(cutset)
    feature_extractor = (
        FeatureExtractor.from_yaml(feature_manifest)
        if feature_manifest is not None
        else Fbank()
    )
    cuts = cuts.compute_and_store_features(
        extractor=feature_extractor,
        storage_path=storage_path,
        num_jobs=num_jobs,
        storage_type=get_writer(storage_type),
    )
    Path(output_cutset).parent.mkdir(parents=True, exist_ok=True)
    cuts.to_file(output_cutset)


@feat.command(context_settings=dict(show_default=True))
@click.argument("cutset", type=click.Path(exists=True, dir_okay=False, allow_dash=True))
@click.argument("output_cutset", type=click.Path(allow_dash=True))
@click.argument("storage_path", type=click.Path())
@click.option(
    "-f",
    "--feature-manifest",
    type=click.Path(exists=True, dir_okay=False),
    help="Optional manifest specifying feature extractor configuration. "
    "If you want to use CUDA, you should specify the device in this "
    "config.",
)
@click.option(
    "--storage-type",
    type=click.Choice(available_storage_backends()),
    default=default_features_storage_backend_name(),
    help="Select a storage backend for the feature matrices. "
    "The default is numpy_files; lilcom_chunky is preferred for smaller "
    "on-disk storage when lilcom is installed. See also: "
    "lhotse list-storage-backends.",
)
@click.option(
    "-j", "--num-jobs", type=int, default=4, help="Number of dataloader workers."
)
@click.option(
    "-b",
    "--batch-duration",
    type=float,
    default=600.0,
    help="At most this many seconds of audio will be processed in each batch.",
)
def extract_cuts_batch(
    cutset: Pathlike,
    output_cutset: Pathlike,
    storage_path: Pathlike,
    feature_manifest: Optional[Pathlike],
    storage_type: str,
    num_jobs: int,
    batch_duration: Seconds,
):
    """
    Extract features for cuts in a given CUTSET manifest.
    The features are stored in STORAGE_PATH, and the output manifest
    with features is stored in OUTPUT_CUTSET.

    This version enables CUDA acceleration for feature extractors
    that support it (e.g., kaldifeat extractors).

    \b
    Example usage of kaldifeat fbank with CUDA:

        $ pip install kaldifeat  # note: ensure it's compiled with CUDA

        $ lhotse feat write-default-config -f kaldifeat-fbank feat.yml

        $ sed 's/device: cpu/device: cuda/' feat.yml feat-cuda.yml

        $ lhotse feat extract-cuts-batch -f feat-cuda.yml cuts.jsonl cuts_with_feats.jsonl feats.h5
    """
    cuts = CutSet.from_file(cutset)
    feature_extractor = (
        FeatureExtractor.from_yaml(feature_manifest)
        if feature_manifest is not None
        else Fbank()
    )
    cuts = cuts.compute_and_store_features_batch(
        extractor=feature_extractor,
        storage_path=storage_path,
        batch_duration=batch_duration,
        num_workers=num_jobs,
        storage_type=get_writer(storage_type),
    )
    Path(output_cutset).parent.mkdir(parents=True, exist_ok=True)
    cuts.to_file(output_cutset)
