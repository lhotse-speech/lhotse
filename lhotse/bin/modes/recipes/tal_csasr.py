import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.tal_csasr import prepare_tal_csasr
from lhotse.utils import Pathlike

__all__ = ["tal_csasr"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
def tal_csasr(corpus_dir: Pathlike, output_dir: Pathlike):
    """Tal_csasr ASR data preparation."""
    prepare_tal_csasr(corpus_dir, output_dir=output_dir)
