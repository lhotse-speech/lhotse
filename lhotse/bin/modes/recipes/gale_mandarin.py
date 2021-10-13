from typing import List, Optional

import click

from lhotse.bin.modes import prepare
from lhotse.recipes.gale_mandarin import prepare_gale_mandarin
from lhotse.utils import Pathlike

__all__ = ["gale_mandarin"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "-s",
    "--audio",
    type=click.Path(exists=True, dir_okay=True),
    multiple=True,
    help="Paths to audio dirs, e.g., LDC2013S08. Multiple corpora can be provided by repeating `-s`.",
)
@click.option(
    "-t",
    "--transcript",
    type=click.Path(exists=True, dir_okay=True),
    multiple=True,
    help="Paths to transcript dirs, e.g., LDC2013T20. Multiple corpora can be provided by repeating `-t`",
)
@click.option(
    "--absolute-paths",
    type=bool,
    default=False,
    help="Use absolute paths for recordings",
)
@click.option(
    "--segment-words",
    type=bool,
    default=False,
    help="Use 'jieba' package to perform word segmentation on the text",
)
def gale_mandarin(
    output_dir: Pathlike,
    audio: Optional[List[Pathlike]] = None,
    transcript: Optional[List[Pathlike]] = None,
    absolute_paths: Optional[bool] = False,
    segment_words: Optional[bool] = False,
):
    """GALE Mandarin Broadcast speech data preparation."""
    prepare_gale_mandarin(
        audio,
        transcript,
        output_dir=output_dir,
        absolute_paths=absolute_paths,
        segment_words=segment_words,
    )
