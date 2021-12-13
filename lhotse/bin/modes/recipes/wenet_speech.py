import click
from lhotse.bin.modes import prepare
from lhotse.recipes.wenet_speech import prepare_wenet_speech
from lhotse.utils import Pathlike

from typing import Sequence


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
    "pass each with `-p` Example: `-p M -p TEST_NET`",
)
@click.option(
    "-j",
    "--num-jobs",
    type=int,
    default=1,
    help="How many threads to use (can give good speed-ups with slow disks).",
)
def wenet_speech(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    dataset_parts: Sequence[str],
    num_jobs: int,
):
    """
    The WenetSpeech corpus preparation.
    """
    prepare_wenet_speech(
        corpus_dir,
        output_dir=output_dir,
        num_jobs=num_jobs,
        dataset_parts=dataset_parts,
    )
