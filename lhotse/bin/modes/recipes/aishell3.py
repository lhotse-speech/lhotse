import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.aishell3 import download_aishell3, prepare_aishell3
from lhotse.utils import Pathlike

__all__ = ["aishell3"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
def aishell3(corpus_dir: Pathlike, output_dir: Pathlike):
    """aishell3 data preparation."""
    prepare_aishell3(corpus_dir, output_dir=output_dir)


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path(), default=".")
def aishell3(target_dir: Pathlike):
    """aishell3 download."""
    download_aishell3(target_dir)
