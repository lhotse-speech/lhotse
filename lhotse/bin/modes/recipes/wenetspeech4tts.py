from typing import Sequence

import click

from lhotse.bin.modes import prepare
from lhotse.recipes import prepare_wenetspeech4tts
from lhotse.utils import Pathlike

__all__ = ["wenetspeech4tts"]


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
    "--dataset-parts",
    type=str,
    default=["all"],
    multiple=True,
    help="List of dataset parts to prepare. To prepare multiple parts, pass each with `-p` "
    "Example: `-p Basic -p Premium`",
)
def wenetspeech4tts(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    dataset_parts: Sequence[str],
    num_jobs: int,
):
    """WenetSpeech4TTS data preparation."""
    prepare_wenetspeech4tts(
        corpus_dir,
        output_dir=output_dir,
        num_jobs=num_jobs,
        dataset_parts=dataset_parts,
    )
