import click

from lhotse.bin.modes import prepare
from lhotse.recipes.mgb2 import prepare_mgb2
from lhotse.utils import Pathlike

__all__ = ["mgb2"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "--text-cleaning/--no-text-cleaning", default=True, help="Basic text cleaning."
)
@click.option(
    "--buck-walter/--no-buck-walter",
    default=False,
    help="Use BuckWalter transliteration.",
)
@click.option(
    "-j",
    "--num-jobs",
    type=int,
    default=1,
    help="How many threads to use (can give good speed-ups with slow disks).",
)
@click.option(
    "--mer-thresh",
    default=80,
    help="filter out segments based on mer (Match Error Rate).",
)
def mgb2(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    text_cleaning: bool,
    buck_walter: bool,
    num_jobs: int,
    mer_thresh: int,
):
    """mgb2 ASR data preparation."""
    prepare_mgb2(
        corpus_dir,
        output_dir,
        text_cleaning=text_cleaning,
        buck_walter=buck_walter,
        num_jobs=num_jobs,
        mer_thresh=mer_thresh,
    )
