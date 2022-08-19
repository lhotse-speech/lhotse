import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.primewords import download_primewords, prepare_primewords
from lhotse.utils import Pathlike

__all__ = ["primewords"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
def primewords(corpus_dir: Pathlike, output_dir: Pathlike):
    """Primewords ASR data preparation."""
    prepare_primewords(corpus_dir, output_dir=output_dir)


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
def primewords(target_dir: Pathlike):
    """Primewords download."""
    download_primewords(target_dir)
