import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes import download_vctk, prepare_vctk
from lhotse.utils import Pathlike

__all__ = ["vctk"]


@prepare.command()
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option("--use-edinburgh-vctk-url", default=False)
def vctk(corpus_dir: Pathlike, output_dir: Pathlike, use_edinburgh_vctk_url: bool):
    """VCTK data preparation."""
    prepare_vctk(
        corpus_dir, output_dir=output_dir, use_edinburgh_vctk_url=use_edinburgh_vctk_url
    )


@download.command()
@click.argument("target_dir", type=click.Path())
def vctk(target_dir: Pathlike):
    """VCTK download."""
    download_vctk(target_dir)
