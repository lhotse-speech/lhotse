from typing import Sequence

import click

from lhotse.bin.modes import prepare
from lhotse.recipes.kespeech import prepare_kespeech
from lhotse.utils import Pathlike


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "-p",
    "--dataset-parts",
    type=str,
    default=["all"],
    multiple=True,
    help="List of dataset parts to prepare. To prepare multiple parts,"
    "pass each with `-p` Example: `-p dev_phase1 -p dev_phase2`",
)
@click.option(
    "-j",
    "--num-jobs",
    type=int,
    default=1,
    help="How many threads to use (can give good speed-ups with slow disks).",
)
def kespeech(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    dataset_parts: Sequence[str],
    num_jobs: int,
):
    """
    The KeSpeech corpus preparation.
    """
    prepare_kespeech(
        corpus_dir,
        output_dir=output_dir,
        num_jobs=num_jobs,
        dataset_parts=dataset_parts,
    )
