from typing import List

import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.tedlium import TEDLIUM_PARTS, download_tedlium, prepare_tedlium
from lhotse.utils import Pathlike


@prepare.command()
@click.argument(
    "tedlium_dir", type=click.Path(exists=True, dir_okay=True, file_okay=False)
)
@click.argument("output_dir", type=click.Path())
@click.option(
    "--parts",
    "-p",
    type=click.Choice(TEDLIUM_PARTS),
    multiple=True,
    default=list(TEDLIUM_PARTS),
    help="Which parts of TED-LIUM v3 to prepare (by default all).",
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
    type=click.Choice(["none", "upper", "kaldi"], case_sensitive=False),
    default="none",
    help="Type of text normalization to apply (no normalization, by default). "
    "Selecting `kaldi` will remove <unk> tokens and join suffixes.",
)
def tedlium(
    tedlium_dir: Pathlike,
    output_dir: Pathlike,
    parts: List[str],
    num_jobs: int,
    normalize_text: str,
):
    """
    TED-LIUM v3 recording and supervision manifest preparation.
    """
    prepare_tedlium(
        tedlium_root=tedlium_dir,
        output_dir=output_dir,
        dataset_parts=parts,
        num_jobs=num_jobs,
        normalize_text=normalize_text,
    )


@download.command()
@click.argument("target_dir", type=click.Path())
def tedlium(target_dir: Pathlike):
    """TED-LIUM v3 download (approx. 11GB)."""
    download_tedlium(target_dir)
