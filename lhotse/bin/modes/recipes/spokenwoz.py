from typing import Sequence

import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes import download_spokenwoz, prepare_spokenwoz
from lhotse.utils import Pathlike

__all__ = ["spokenwoz"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "-j",
    "--num-jobs",
    type=int,
    default=1,
    help="How many jobs to use (can give good speed-ups with slow disks).",
)
@click.option(
    "-p",
    "--dataset-splits",
    type=str,
    default=["all"],
    multiple=True,
    help="List of dataset parts to prepare. To prepare multiple parts, pass each with `-p` "
    "Example: `-p train -p dev -p test`",
)
def spokenwoz(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    dataset_splits: Sequence[str],
    num_jobs: int,
):
    """SpokenWOZ data preparation."""
    prepare_spokenwoz(
        corpus_dir,
        output_dir=output_dir,
        num_jobs=num_jobs,
        dataset_splits=dataset_splits,
    )


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
@click.option(
    "-p",
    "--dataset-parts",
    type=str,
    default=["all"],
    multiple=True,
    help="List of dataset parts to download. To prepare multiple parts, pass each with `-p` "
    "Example: `-p train_dev -p test`",
)
def spokenwoz(
    target_dir: Pathlike,
    dataset_parts: Sequence[str],
):
    """SpokenWOZ data download."""
    download_spokenwoz(target_dir, dataset_parts=dataset_parts)
