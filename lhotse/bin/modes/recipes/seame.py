from typing import Optional, Sequence, Union

import click

from lhotse.bin.modes import prepare
from lhotse.recipes.seame import prepare_seame
from lhotse.utils import Pathlike


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("split_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "--clean-text",
    default=False,
    help="Whether to perform additional text cleaning and normalization",
)
@click.option(
    "--delimiter",
    default="",
    help="Used to split the code switching text ",
)
def seame(
    corpus_dir: Pathlike,
    split_dir: Pathlike,
    clean_text: bool,
    delimiter: str,
    output_dir: Pathlike,
):
    """
    SEAME data preparation.
    \b
    This is Singaporean Codeswitched English and Mandarin data.
    """
    prepare_seame(
        corpus_dir=corpus_dir,
        split_dir=split_dir,
        output_dir=output_dir,
        clean_text=clean_text,
        delimiter=delimiter,
    )
