from typing import List, Optional, Sequence, Tuple, Union

import click

from lhotse.bin.modes import prepare
from lhotse.recipes.radio import prepare_radio
from lhotse.utils import Pathlike

__all__ = ["radio"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(dir_okay=True))
@click.argument("output_dir", type=click.Path(dir_okay=True))
@click.option(
    "-d",
    "--min-seg-dur",
    type=float,
    default=0.5,
    help="The minimum segment duration",
)
@click.option(
    "-j",
    "--num-jobs",
    type=int,
    default=4,
    help="The number of parallel threads to use for data preparation",
)
def radio(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    min_seg_dur: float = 0.5,
    num_jobs: int = 4,
):
    """Data preparation"""
    prepare_radio(
        corpus_dir,
        output_dir=output_dir,
        num_jobs=num_jobs,
        min_segment_duration=min_seg_dur,
    )
