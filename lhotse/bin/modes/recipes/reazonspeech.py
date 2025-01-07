import logging
from typing import List

import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.reazonspeech import (
    REAZONSPEECH,
    download_reazonspeech,
    prepare_reazonspeech,
)
from lhotse.utils import Pathlike

__all__ = ["reazonspeech"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "-j",
    "--num-jobs",
    type=int,
    default=1,
    help="How many threads to use (can give good speed-ups with slow disks).",
)
def reazonspeech(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    num_jobs: int,
):
    """ReazonSpeech ASR data preparation."""
    logging.basicConfig(level=logging.INFO)
    prepare_reazonspeech(corpus_dir, output_dir=output_dir, num_jobs=num_jobs)


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
@click.option(
    "--subset",
    type=click.Choice(("auto",) + REAZONSPEECH),
    multiple=True,
    default=["auto"],
    help="List of dataset parts to prepare (default: small-v1). To prepare multiple parts, pass each with `--subset` "
    "Example: `--subset all",
)
@click.option(
    "-j",
    "--num-jobs",
    type=int,
    default=1,
    help="How many threads to use (can give good speed-ups with slow disks).",
)
def reazonspeech(target_dir: Pathlike, subset: List[str], num_jobs: int):
    """ReazonSpeech download."""
    logging.basicConfig(level=logging.INFO)
    if "auto" in subset:
        subset = "auto"
    download_reazonspeech(target_dir, dataset_parts=subset, num_jobs=num_jobs)
