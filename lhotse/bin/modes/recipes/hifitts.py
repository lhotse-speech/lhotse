import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes import download_hifitts, prepare_hifitts
from lhotse.utils import Pathlike


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
def hifitts(corpus_dir: Pathlike, output_dir: Pathlike, num_jobs: int):
    """HiFiTTS data preparation."""
    prepare_hifitts(corpus_dir, output_dir=output_dir, num_jobs=num_jobs)


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
def hifitts(
    target_dir: Pathlike,
):
    """HiFiTTS data download."""
    download_hifitts(target_dir)
