import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.magicdata import download_magicdata, prepare_magicdata
from lhotse.utils import Pathlike

__all__ = ["magicdata"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
def magicdata(corpus_dir: Pathlike, output_dir: Pathlike):
    """Magicdata ASR data preparation."""
    prepare_magicdata(corpus_dir, output_dir=output_dir)


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
def magicdata(target_dir: Pathlike):
    """Magicdata download."""
    download_magicdata(target_dir)
