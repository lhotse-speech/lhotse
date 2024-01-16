from typing import Optional, Sequence

import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.sbcsae import download_sbcsae, prepare_sbcsae
from lhotse.utils import Pathlike

__all__ = ["sbcsae"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
def sbcsae(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
):
    """SBCSAE data preparation."""
    prepare_sbcsae(corpus_dir, output_dir=output_dir)


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
@click.option(
    "--download-mp3",
    type=bool,
    is_flag=True,
    default=False,
    help="Download the mp3 copy of the audio as well as wav.",
)
def sbcsae(
    target_dir: Pathlike,
    download_mp3: Optional[bool] = False,
):
    """SBCSAE download."""
    download_sbcsae(target_dir, download_mp3=download_mp3)
