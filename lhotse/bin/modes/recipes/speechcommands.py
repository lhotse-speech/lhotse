from typing import Dict, List, Optional, Tuple, Union

import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.speechcommands import (
    download_speechcommands,
    prepare_speechcommands,
)
from lhotse.utils import Pathlike


@prepare.command(context_settings=dict(show_default=True))
@click.argument("speechcommands_version", type=str)
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
def speechcommands(
    speechcommands_version: str,
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
):
    """Speech Commands v0.01 or v0.02 data preparation."""
    prepare_speechcommands(
        speechcommands_version=speechcommands_version,
        corpus_dir=corpus_dir,
        output_dir=output_dir,
    )


@download.command(context_settings=dict(show_default=True))
@click.argument("speechcommands_version", type=str)
@click.argument("target_dir", type=click.Path())
@click.option("--force-download", is_flag=True, default=False, help="Force download")
def speechcommands(
    speechcommands_version: str,
    target_dir: Pathlike,
    force_download: Optional[bool] = False,
):
    """Speech Commands v0.01 or v0.02 download."""
    download_speechcommands(
        speechcommands_version=speechcommands_version,
        target_dir=target_dir,
        force_download=force_download,
    )
