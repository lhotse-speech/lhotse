import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.aishell4 import download_aishell4, prepare_aishell4
from lhotse.utils import Pathlike

__all__ = ["aishell4"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
def aishell4(corpus_dir: Pathlike, output_dir: Pathlike):
    """AISHELL-4 data preparation."""
    prepare_aishell4(corpus_dir, output_dir=output_dir)


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
def aishell4(target_dir: Pathlike):
    """AISHELL-4 download."""
    download_aishell4(target_dir)
