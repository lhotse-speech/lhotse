from typing import Sequence, Union

import click

from lhotse.bin.modes import prepare
from lhotse.recipes.csj import prepare_csj
from lhotse.utils import Pathlike

__all__ = ["csj"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("manifest_dir", type=click.Path())
@click.option(
    "-t",
    "--transcript-dir",
    type=click.Path(),
    default=None,
    help=(
        "Directory to save parsed transcripts in txt format, with "
        "valid and eval sets created from the core and noncore datasets. "
        "If not provided, this script will not create valid and eval "
        "sets."
    ),
)
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
    corpus_dir: Pathlike,
    manifest_dir: Pathlike,
    dataset_parts: Union[str, Sequence[str]],
    transcript_dir: Pathlike,
    num_jobs: int,
):
    "Prepare Corpus of Spontaneous Japanese"

    prepare_csj(
        corpus_dir=corpus_dir,
        manifest_dir=manifest_dir,
        dataset_parts=dataset_parts,
        transcript_dir=transcript_dir,
        nj=num_jobs,
    )
