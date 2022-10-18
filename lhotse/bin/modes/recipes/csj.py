from typing import Sequence, Union

import click

from lhotse.bin.modes import prepare
from lhotse.recipes.csj import prepare_csj
from lhotse.utils import Pathlike

__all__ = ["csj"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("transcript_dir", type=click.Path())
@click.argument("manifest_dir", type=click.Path())
@click.option(
    "-c",
    "--configs",
    type=click.Path(exists=True, file_okay=True),
    default=None,
    multiple=True,
    help=(
        "List of .ini config files to use when parsing transcripts. "
        "Pass each with `-c`. Example: "
        "`-c local/config/disfluent.ini -c local/config/fluent.ini` "
        "NOTE: Only the segment specifications in the first config file are "
        "used. The remaining config files that come after will inherit "
        "the segment boundaries of the first config file. "
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
    transcript_dir: Pathlike,
    manifest_dir: Pathlike,
    configs: Union[Pathlike, Sequence[Pathlike]],
    dataset_parts: Union[str, Sequence[str]],
    num_jobs: int,
):
    "Prepare Corpus of Spontaneous Japanese"

    prepare_csj(
        corpus_dir=corpus_dir,
        transcript_dir=transcript_dir,
        manifest_dir=manifest_dir,
        dataset_parts=dataset_parts,
        configs=configs,
        nj=num_jobs,
    )
