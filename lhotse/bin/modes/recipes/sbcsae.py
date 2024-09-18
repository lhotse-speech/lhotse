from typing import Optional, Sequence

import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.sbcsae import download_sbcsae, prepare_sbcsae
from lhotse.utils import Pathlike

__all__ = ["sbcsae"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "--geolocation",
    type=bool,
    is_flag=True,
    default=False,
    help="Include geographic coordinates of speakers' hometowns in the manifests.",
)
@click.option(
    "--omit-realignments",
    type=bool,
    is_flag=True,
    default=False,
    help="Only output the original corpus segmentation without boundary improvements.",
)
def sbcsae(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    geolocation: bool,
    omit_realignments: bool,
):
    """SBCSAE data preparation."""
    prepare_sbcsae(
        corpus_dir,
        output_dir=output_dir,
        geolocation=geolocation,
        omit_realignments=omit_realignments,
    )


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
@click.option(
    "--force-download",
    type=bool,
    is_flag=True,
    default=False,
    help="Force download.",
)
def sbcsae(
    target_dir: Pathlike,
    force_download: bool,
):
    """SBCSAE download."""
    download_sbcsae(target_dir, force_download=force_download)
