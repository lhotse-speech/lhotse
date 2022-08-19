import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.thchs_30 import download_thchs_30, prepare_thchs_30
from lhotse.utils import Pathlike

__all__ = ["thchs_30"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
def thchs_30(corpus_dir: Pathlike, output_dir: Pathlike):
    """thchs_30 ASR data preparation."""
    prepare_thchs_30(corpus_dir, output_dir=output_dir)


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
def thchs_30(target_dir: Pathlike):
    """thchs_30 download."""
    download_thchs_30(target_dir)
