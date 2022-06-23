import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.stcmds import download_stcmds, prepare_stcmds
from lhotse.utils import Pathlike

__all__ = ["stcmds"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
def stcmds(corpus_dir: Pathlike, output_dir: Pathlike):
    """Stcmds ASR data preparation."""
    prepare_stcmds(corpus_dir, output_dir=output_dir)


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
def stcmds(target_dir: Pathlike):
    """Stcmds download."""
    download_stcmds(target_dir)
