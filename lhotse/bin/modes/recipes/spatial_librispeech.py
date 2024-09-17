from typing import Sequence

import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.spatial_librispeech import (
    download_spatial_librispeech,
    prepare_spatial_librispeech,
)
from lhotse.utils import Pathlike

__all__ = ["spatial_librispeech"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "-p",
    "--dataset-parts",
    type=str,
    default=["all"],
    multiple=True,
    help="List of dataset parts to prepare. To prepare multiple parts, pass each with `-p` "
    "Example: `-p train -p test`",
)
@click.option(
    "-j",
    "--num-jobs",
    type=int,
    default=1,
    help="How many threads to use (can give good speed-ups with slow disks).",
)
@click.option(
    "--normalize-text",
    type=click.Choice(["none", "lower"], case_sensitive=False),
    default="none",
    help="Conversion of transcripts to lower-case (originally in upper-case).",
    show_default=True,
)
def spatial_librispeech(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    dataset_parts: Sequence[str],
    normalize_text: str,
    num_jobs: int,
):
    """Spatial-LibriSpeech ASR data preparation."""
    if len(dataset_parts) == 1:
        dataset_parts = dataset_parts[0]
    prepare_spatial_librispeech(
        corpus_dir,
        output_dir=output_dir,
        dataset_parts=dataset_parts,
        normalize_text=normalize_text,
        num_jobs=num_jobs,
    )


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
@click.option(
    "-p",
    "--dataset-parts",
    type=str,
    default=["all"],
    multiple=True,
    help="List of dataset parts to download. To prepare multiple parts, pass each with `-p` "
    "Example: `-p train -p test`",
)
@click.option(
    "-j",
    "--num-jobs",
    type=int,
    default=1,
    help="How many threads to use (can give good speed-ups with slow disks).",
)
def spatial_librispeech(
    target_dir: Pathlike,
    dataset_parts: Sequence[str],
    num_jobs: int,
):
    """Spatial-LibriSpeech download."""
    if len(dataset_parts) == 1:
        dataset_parts = dataset_parts[0]
    download_spatial_librispeech(
        target_dir, dataset_parts=dataset_parts, num_jobs=num_jobs
    )
