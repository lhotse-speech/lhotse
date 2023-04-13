from typing import Sequence, Union

import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes import download_but_reverb_db, prepare_but_reverb_db
from lhotse.utils import Pathlike

__all__ = ["but_reverb_db"]


@prepare.command()
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "--parts",
    "-p",
    type=str,
    multiple=True,
    default=["silence", "rir"],
    show_default=True,
    help="Parts to prepare.",
)
def but_reverb_db(
    corpus_dir: Pathlike, output_dir: Pathlike, parts: Union[str, Sequence[str]]
):
    """BUT Reverb DB data preparation."""
    prepare_but_reverb_db(corpus_dir, output_dir=output_dir, parts=parts)


@download.command()
@click.argument("target_dir", type=click.Path())
@click.option(
    "--force-download",
    type=bool,
    default=False,
    help="If True, download even if file is present.",
)
def but_reverb_db(target_dir: Pathlike, force_download: bool):
    """BUT Reverb DB download."""
    download_but_reverb_db(target_dir, force_download=force_download)
