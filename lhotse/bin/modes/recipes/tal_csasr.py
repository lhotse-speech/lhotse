import click

from lhotse.bin.modes import prepare
from lhotse.recipes.tal_csasr import prepare_tal_csasr
from lhotse.utils import Pathlike

__all__ = ["tal_csasr"]


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
def tal_csasr(corpus_dir: Pathlike, output_dir: Pathlike, num_jobs: int):
    """Tal_csasr ASR data preparation."""
    prepare_tal_csasr(corpus_dir, output_dir=output_dir, num_jobs=num_jobs)
