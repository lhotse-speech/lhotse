import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.ljspeech import download_ljspeech, prepare_ljspeech
from lhotse.utils import Pathlike

__all__ = ["ljspeech"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
def ljspeech(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
):
    """LJSpeech data preparation."""
    prepare_ljspeech(corpus_dir, output_dir=output_dir)


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path(), default=".")
def ljspeech(target_dir: Pathlike):
    """LJSpeech download."""
    download_ljspeech(target_dir)
