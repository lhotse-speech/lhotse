import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.himia import download_himia, prepare_himia
from lhotse.utils import Pathlike

__all__ = ["himia"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
def himia(corpus_dir: Pathlike, output_dir: Pathlike):
    """HI_MIA and HI_MIA_CW data preparation."""
    prepare_himia(corpus_dir, output_dir=output_dir)


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
def himia(target_dir: Pathlike):
    """HI-MIA and HI_MIA_CW download."""
    download_himia(target_dir)
