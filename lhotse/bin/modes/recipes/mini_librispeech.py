import click

from lhotse.bin.modes import prepare, obtain
from lhotse.recipes.mini_librispeech import prepare_mini_librispeech, download_and_untar
from lhotse.utils import Pathlike

__all__ = ['mini_librispeech']


@prepare.command(context_settings=dict(show_default=True))
@click.argument('corpus_dir', type=click.Path(exists=True, dir_okay=True))
@click.argument('output_dir', type=click.Path())
def mini_librispeech(
        corpus_dir: Pathlike,
        output_dir: Pathlike
):
    """Recipe to prepare the manifests for MiniLibriSpeech task."""
    prepare_mini_librispeech(corpus_dir, output_dir)


@obtain.command(context_settings=dict(show_default=True))
@click.argument('target_dir', type=click.Path())
def mini_librispeech(
        target_dir: Pathlike
):
    """Obtain MiniLibriSpeech dataset."""
    download_and_untar(target_dir)
