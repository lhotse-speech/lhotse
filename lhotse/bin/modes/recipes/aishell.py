import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.aishell import download_aishell, prepare_aishell
from lhotse.utils import Pathlike

__all__ = ["aishell"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
def aishell(corpus_dir: Pathlike, output_dir: Pathlike):
    """Aishell ASR data preparation."""
    prepare_aishell(corpus_dir, output_dir=output_dir)


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
def aishell(target_dir: Pathlike):
    """Aishell download."""
    download_aishell(target_dir)
