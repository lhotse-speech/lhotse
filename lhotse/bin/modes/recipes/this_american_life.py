import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.this_american_life import (
    download_this_american_life,
    prepare_this_american_life,
)
from lhotse.utils import Pathlike

__all__ = ["this_american_life"]


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
@click.option(
    "-f",
    "--force-download",
    is_flag=True,
    default=False,
)
def this_american_life(target_dir: Pathlike, force_download: bool = False):
    """This American Life dataset download."""
    download_this_american_life(target_dir)


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
def this_american_life(corpus_dir: Pathlike, output_dir: Pathlike):
    """This American Life data preparation."""
    prepare_this_american_life(corpus_dir, output_dir=output_dir)
