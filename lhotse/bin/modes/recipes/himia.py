from typing import Sequence

import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.himia import download_himia, prepare_himia
from lhotse.utils import Pathlike

__all__ = ["himia"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "-p",
    "--dataset-parts",
    type=str,
    default=["auto"],
    multiple=True,
    help="List of dataset parts to prepare. To prepare multiple parts, pass each with `-p` "
    "Example: `-p test -p cw_test` "
    "Prepare both HI_MIA and HI_MIA_CW by default "
    "All possible data parts are train, dev, test and cw_test",
)
def himia(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    dataset_parts: Sequence[str],
):
    """HI_MIA and HI_MIA_CW data preparation."""
    if len(dataset_parts) == 1:
        dataset_parts = dataset_parts[0]
    prepare_himia(
        corpus_dir=corpus_dir,
        output_dir=output_dir,
        dataset_parts=dataset_parts,
    )


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
@click.option(
    "-p",
    "--dataset-parts",
    type=str,
    default=["auto"],
    multiple=True,
    help="List of dataset parts to download. To download multiple parts, pass each with `-p` "
    "Example: `-p test -p cw_test` "
    "Download both HI_MIA and HI_MIA_CW by default "
    "All possible data parts are train, dev, test and cw_test",
)
def himia(
    target_dir: Pathlike,
    dataset_parts: Sequence[str],
):
    """HI-MIA and HI_MIA_CW download."""
    if len(dataset_parts) == 1:
        dataset_parts = dataset_parts[0]
    download_himia(target_dir, dataset_parts=dataset_parts)
