import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.librispeech import download_librispeech, prepare_librispeech
from lhotse.utils import Pathlike

__all__ = ['librispeech']


@prepare.command(context_settings=dict(show_default=True))
@click.argument('corpus_dir', type=click.Path(exists=True, dir_okay=True))
@click.argument('output_dir', type=click.Path())
@click.option('-j', '--num-jobs', type=int, default=1,
              help='How many threads to use (can give good speed-ups with slow disks).')
def librispeech(
        corpus_dir: Pathlike,
        output_dir: Pathlike,
        num_jobs: int
):
    """(Mini) Librispeech ASR data preparation."""
    prepare_librispeech(corpus_dir, output_dir=output_dir, num_jobs=num_jobs)


@download.command(context_settings=dict(show_default=True))
@click.argument('target_dir', type=click.Path())
@click.option('--full/--mini', default=True, help='Download Librispeech [default] or mini Librispeech.')
def librispeech(
        target_dir: Pathlike,
        full: bool
):
    """(Mini) Librispeech download."""
    download_librispeech(target_dir, dataset_parts='librispeech' if full else 'mini_librispeech')
