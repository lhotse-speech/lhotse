from typing import Optional, Sequence, Union

import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.iwslt2022_dialect import (
    prepare_iwslt2022_dialect,
    prepare_iwslt2022_dialect_eval,
)
from lhotse.utils import Pathlike


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("splits", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "-j",
    "--num-jobs",
    type=int,
    default=1,
    help="How many threads to use (can give good speed-ups with slow disks).",
)
@click.option(
    "--normalized",
    default=False,
    help="Whether to perform additional text normalization.",
)
def iwslt2022_dialect(
    corpus_dir: Pathlike,
    splits: Pathlike,
    output_dir: Pathlike,
    normalized: bool,
    num_jobs: int,
):
    """
    IWSLT_2022 data preparation.

    \b
    This is conversational telephone speech collected as 8kHz-sampled data.
    The catalog number LDC2022E01 corresponds to the train, dev, and test1
    splits of the iwslt2022 shared task.

    To obtaining this data your institution needs to have an LDC subscription.
    You also should download the predined splits with

    git clone https://github.com/kevinduh/iwslt22-dialect.git
    """
    prepare_iwslt2022_dialect(
        corpus_dir,
        splits,
        output_dir=output_dir,
        num_jobs=num_jobs,
        cleaned=normalized,
    )
