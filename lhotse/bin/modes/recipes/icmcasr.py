from typing import Dict, List, Optional, Tuple, Union

import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.icmcasr import prepare_icmcasr
from lhotse.utils import Pathlike


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "-j",
    "--num-jobs",
    type=int,
    default=1,
    help="How many threads to use (can give good speed-ups with slow disks).",
)
@click.option(
    "--mic",
    type=click.Choice(["ihm", "sdm", "mdm"]),
    default="ihm",
    help="Microphone type.",
)
def icmcasr(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    mic: str = "ihm",
    num_jobs: int = 1,
):
    """ICMC-ASR data preparation."""
    prepare_icmcasr(
        corpus_dir=corpus_dir,
        output_dir=output_dir,
        mic=mic,
        num_jobs=num_jobs,
    )
