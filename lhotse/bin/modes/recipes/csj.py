from typing import Sequence, Union

import click

from lhotse.bin.modes import prepare
from lhotse.recipes.csj import prepare_csj
from lhotse.utils import Pathlike

__all__ = ["csj"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("transcript_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "-p",
    "--dataset-parts",
    type=str,
    default=None,
    multiple=True,
    help=(
        "List of dataset parts to prepare. "
        "To prepare multiple parts, pass each with `-p` "
        "Example: `-p eval1 -p eval2`"
    ),
)
@click.option(
    "-j",
    "--num-jobs",
    type=int,
    default=1,
    help="How many threads to use (can give good speed-ups with slow disks).",
)
def csj(
    transcript_dir: Pathlike,
    output_dir: Pathlike,
    dataset_parts: Union[str, Sequence[str]],
    num_jobs: int,
):
    prepare_csj(
        transcript_dir=transcript_dir,
        output_dir=output_dir,
        num_jobs=num_jobs,
        dataset_parts=dataset_parts,
    )
