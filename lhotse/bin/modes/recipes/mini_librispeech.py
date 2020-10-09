import click

from lhotse.bin.modes import obtain, prepare
from lhotse.recipes.librispeech import download_and_untar, prepare_librispeech
from lhotse.utils import Pathlike

__all__ = ['mini_librispeech']


@prepare.command(context_settings=dict(show_default=True))
@click.argument('corpus_dir', type=click.Path(exists=True, dir_okay=True))
@click.argument('output_dir', type=click.Path())
def mini_librispeech(
        corpus_dir: Pathlike,
        output_dir: Pathlike
):
    """Mini Librispeech ASR data preparation."""
    prepare_librispeech(corpus_dir, output_dir=output_dir)


@obtain.command(context_settings=dict(show_default=True))
@click.argument('target_dir', type=click.Path())
def mini_librispeech(
        target_dir: Pathlike
):
    """Mini Librispeech download."""
    download_and_untar(target_dir)
