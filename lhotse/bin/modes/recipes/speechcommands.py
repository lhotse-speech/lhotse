from typing import Sequence

import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.speechcommands import (
    download_speechcommands1,
    download_speechcommands2,
    prepare_speechcommands,
)
from lhotse.utils import Pathlike


@prepare.command(context_settings=dict(show_default=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "--speechcommands1",
    "-v1",
    type=click.Path(),
    default=None,
    help="Path to Speech Commands v0.01 dataset.",
)
@click.option(
    "--speechcommands2",
    "-v2",
    type=click.Path(),
    default=None,
    help="Path to Speech Commands v0.02 dataset.",
)
def speechcommands(
    speechcommands1: Optional[Pathlike],
    speechcommands2: Optional[Pathlike],
    output_dir: Pathlike,
):
    """Speech Commands v1 or v2 data preparation."""
    prepare_speechcommands(
        speechcommands1_root=speechcommands1,
        speechcommands2_root=speechcommands2,
        output_dir=output_dir,
    )


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
@click.option("--force-download", is_flag=True, default=False, help="Force download")
def speechcommands1(target_dir: Pathlike, force_download: Optional[bool] = False):
    """Speech Commands v0.01 download."""
    download_speechcommands1(target_dir, force_download=force_download)


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
@click.option("--force-download", is_flag=True, default=False, help="Force download")
def speechcommands2(target_dir: Pathlike, force_download: Optional[bool] = False):
    """Speech Commands v0.02 download."""
    download_speechcommands2(target_dir, force_download=force_download)
