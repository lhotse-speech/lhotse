import click

from lhotse.bin.modes import prepare
from lhotse.recipes.cdsd import prepare_cdsd
from lhotse.utils import Pathlike

__all__ = ["cdsd"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
def cdsd(corpus_dir: Pathlike, output_dir: Pathlike):
    """CDSD ASR data preparation."""
    prepare_cdsd(corpus_dir, output_dir=output_dir)
