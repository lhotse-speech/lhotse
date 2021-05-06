import click

from typing import Optional, List

from lhotse.bin.modes import prepare
from lhotse.recipes.cmu_kids import prepare_cmu_kids
from lhotse.utils import Pathlike

__all__ = ["cmu_kids"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "--absolute-paths",
    type=bool,
    default=True,
    help="Use absolute paths for recordings",
)
def cmu_kids(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    absolute_paths: Optional[bool] = False,
):
    """CMU Kids corpus data preparation."""
    prepare_cmu_kids(
        corpus_dir,
        output_dir=output_dir,
        absolute_paths=absolute_paths,
    )
