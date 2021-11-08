from typing import List, Optional

import click

from lhotse.bin.modes import prepare
from lhotse.recipes.gale_arabic import prepare_gale_arabic
from lhotse.utils import Pathlike

__all__ = ["gale_arabic"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "-s",
    "--audio",
    type=click.Path(exists=True, dir_okay=True),
    multiple=True,
    help="Paths to audio dirs, e.g., LDC2013S02. Multiple corpora can be provided by repeating `-s`.",
)
@click.option(
    "-t",
    "--transcript",
    type=click.Path(exists=True, dir_okay=True),
    multiple=True,
    help="Paths to transcript dirs, e.g., LDC2013T17. Multiple corpora can be provided by repeating `-t`",
)
@click.option(
    "--absolute-paths",
    type=bool,
    default=False,
    help="Use absolute paths for recordings",
)
def gale_arabic(
    output_dir: Pathlike,
    audio: Optional[List[Pathlike]] = None,
    transcript: Optional[List[Pathlike]] = None,
    absolute_paths: Optional[bool] = False,
):
    """GALE Arabic Phases 1 to 4 Broadcast news and conversation data preparation."""
    prepare_gale_arabic(
        audio,
        transcript,
        output_dir=output_dir,
        absolute_paths=absolute_paths,
    )
