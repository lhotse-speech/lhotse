import logging
from typing import List

import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.gigaspeech import (
    GIGASPEECH_PARTS,
    download_gigaspeech,
    prepare_gigaspeech,
)
from lhotse.utils import Pathlike


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "--subset",
    type=click.Choice(("auto",) + GIGASPEECH_PARTS),
    multiple=True,
    default=["auto"],
    help="Which parts of Gigaspeech to download (by default XL + DEV + TEST).",
)
@click.option(
    "-j",
    "--num-jobs",
    type=int,
    default=1,
    help="How many threads to use (can give good speed-ups with slow disks).",
)
def gigaspeech(
    corpus_dir: Pathlike, output_dir: Pathlike, subset: List[str], num_jobs: int
):
    """Gigaspeech ASR data preparation."""
    logging.basicConfig(level=logging.INFO)
    if "auto" in subset:
        subset = "auto"
    prepare_gigaspeech(
        corpus_dir, output_dir=output_dir, dataset_parts=subset, num_jobs=num_jobs
    )


@download.command(context_settings=dict(show_default=True))
@click.argument("password", type=str)
@click.argument("target_dir", type=click.Path())
@click.option(
    "--subset",
    type=click.Choice(("auto",) + GIGASPEECH_PARTS),
    multiple=True,
    default=["auto"],
    help="Which parts of Gigaspeech to download (by default XL + DEV + TEST).",
)
@click.option(
    "--host", type=str, default="tsinghua", help="Which host to download Gigaspeech."
)
def gigaspeech(password: str, target_dir: Pathlike, subset: List[str], host: str):
    """Gigaspeech download."""
    # Convert (likely one-element) list with "auto" into a string.
    logging.basicConfig(level=logging.INFO)
    if "auto" in subset:
        subset = "auto"
    download_gigaspeech(password, target_dir, dataset_parts=subset, host=host)
