import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes import download_libritts, prepare_libritts
from lhotse.utils import Pathlike

__all__ = ['libritts']


@prepare.command(context_settings=dict(show_default=True))
@click.argument('corpus_dir', type=click.Path(exists=True, dir_okay=True))
@click.argument('output_dir', type=click.Path())
@click.option('-j', '--num-jobs', type=int, default=1,
              help='How many jobs to use (can give good speed-ups with slow disks).')
def libritts(
        corpus_dir: Pathlike,
        output_dir: Pathlike,
        num_jobs: int
):
    """LibriTTs data preparation."""
    prepare_libritts(corpus_dir, output_dir=output_dir, num_jobs=num_jobs)


@download.command(context_settings=dict(show_default=True))
@click.argument('target_dir', type=click.Path())
def libritts(
        target_dir: Pathlike,
):
    """LibriTTS data download."""
    download_libritts(target_dir)
