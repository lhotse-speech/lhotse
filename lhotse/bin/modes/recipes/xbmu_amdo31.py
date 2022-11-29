import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.xbmu_amdo31 import download_xbmu_amdo31, prepare_xbmu_amdo31
from lhotse.utils import Pathlike

__all__ = ["xbmu_amdo31"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
def xbmu_amdo31(corpus_dir: Pathlike, output_dir: Pathlike):
    """XBMU-AMDO31 ASR data preparation."""
    prepare_xbmu_amdo31(corpus_dir, output_dir=output_dir)


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
def xbmu_amdo31(target_dir: Pathlike):
    """XBMU-AMDO31 download."""
    download_xbmu_amdo31(target_dir)
