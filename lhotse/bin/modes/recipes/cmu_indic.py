import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.cmu_indic import download_cmu_indic, prepare_cmu_indic
from lhotse.utils import Pathlike


@prepare.command()
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
def cmu_indic(corpus_dir: Pathlike, output_dir: Pathlike):
    """CMU Indic data preparation."""
    prepare_cmu_indic(corpus_dir, output_dir=output_dir)


@download.command()
@click.argument("target_dir", type=click.Path())
def cmu_indic(target_dir: Pathlike):
    """CMU Indic download."""
    download_cmu_indic(target_dir)
