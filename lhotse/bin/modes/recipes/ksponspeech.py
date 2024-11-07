from typing import Sequence

import click

from lhotse.bin.modes import prepare
from lhotse.recipes.ksponspeech import prepare_ksponspeech
from lhotse.utils import Pathlike

__all__ = ["ksponspeech"]


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
    type=click.Choice(["none", "default"], case_sensitive=False),
    default="default",
    help="Type of text normalization to apply.",
)
def ksponspeech(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    dataset_parts: Sequence[str],
    num_jobs: int,
    normalize_text: str,
):
    """KsponSpeech ASR data preparation."""
    if len(dataset_parts) == 1:
        dataset_parts = dataset_parts[0]
    prepare_ksponspeech(
        corpus_dir,
        output_dir=output_dir,
        num_jobs=num_jobs,
        dataset_parts=dataset_parts,
        normalize_text=normalize_text,
    )
