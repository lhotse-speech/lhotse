import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.cmu_arctic import download_cmu_arctic, prepare_cmu_arctic
from lhotse.utils import Pathlike

__all__ = ["cmu_arctic"]


@prepare.command()
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
def cmu_arctic(corpus_dir: Pathlike, output_dir: Pathlike):
    """CMU Arctic data preparation."""
    prepare_cmu_arctic(corpus_dir, output_dir=output_dir)


@download.command()
@click.argument("target_dir", type=click.Path())
def cmu_arctic(target_dir: Pathlike):
    """CMU Arctic download."""
    download_cmu_arctic(target_dir)
