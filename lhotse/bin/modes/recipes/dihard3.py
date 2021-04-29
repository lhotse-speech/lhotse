import click

from typing import Optional

from lhotse.bin.modes import prepare
from lhotse.recipes.dihard3 import prepare_dihard3
from lhotse.utils import Pathlike

__all__ = ["dihard3"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("output_dir", type=click.Path())
@click.option("--dev", type=click.Path(exists=True, dir_okay=True))
@click.option("--eval", type=click.Path(exists=True, dir_okay=True))
@click.option(
    "--uem/--no-uem",
    default=True,
    help="Specify whether or not to create UEM supervision",
)
@click.option(
    "-j",
    "--num-jobs",
    type=int,
    default=1,
    help="Number of jobs to scan corpus directory for recordings.",
)
def dihard3(
    output_dir: Pathlike,
    dev: Optional[Pathlike],
    eval: Optional[Pathlike],
    uem: Optional[float] = True,
    num_jobs: Optional[int] = 1,
):
    """DIHARD3 data preparation."""
    prepare_dihard3(
        dev, eval, output_dir=output_dir, uem_manifest=uem, num_jobs=num_jobs
    )
