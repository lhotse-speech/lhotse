import click

from typing import Optional

from lhotse.bin.modes import prepare
from lhotse.recipes.cslu_kids import prepare_cslu_kids
from lhotse.utils import Pathlike

__all__ = ["cslu_kids"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "--absolute-paths",
    type=bool,
    default=True,
    help="Use absolute paths for recordings",
)
@click.option(
    "--normalize-text",
    type=bool,
    default=True,
    help="Remove noise tags (<bn>, <bs>) from spontaneous speech transcripts",
)
def cslu_kids(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    absolute_paths: Optional[bool] = False,
    normalize_text: Optional[bool] = True,
):
    """CSLU Kids corpus data preparation."""
    prepare_cslu_kids(
        corpus_dir,
        output_dir=output_dir,
        absolute_paths=absolute_paths,
        normalize_text=normalize_text,
    )
