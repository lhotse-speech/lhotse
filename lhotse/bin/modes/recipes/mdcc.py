from typing import Optional, Sequence

import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.mdcc import download_mdcc, prepare_mdcc
from lhotse.utils import Pathlike


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
    "Example: `-p train -p valid`",
)
def MDCC(
    corpus_dir: Pathlike,
    dataset_parts: Sequence[str],
    output_dir: Optional[Pathlike] = None,
):
    """MDCC data preparation."""
    prepare_mdcc(
        corpus_dir=corpus_dir,
        dataset_parts=dataset_parts,
        output_dir=output_dir,
    )


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
@click.option(
    "--force-download",
    is_flag=True,
    default=False,
    help="if True, it will download the MDCC data even if it is already present.",
)
def MDCC(
    target_dir: Pathlike,
    force_download: Optional[bool] = False,
):
    """MDCC download."""
    download_mdcc(
        target_dir=target_dir,
        force_download=force_download,
    )
