import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.wham import download_wham, prepare_wham
from lhotse.utils import Pathlike

__all__ = ["wham"]


@prepare.command()
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
def wham(corpus_dir: Pathlike, output_dir: Pathlike):
    """WHAM data preparation."""
    prepare_wham(corpus_dir, output_dir=output_dir)


@download.command()
@click.argument("target_dir", type=click.Path())
def wham(target_dir: Pathlike):
    """WHAM download."""
    download_wham(target_dir)
