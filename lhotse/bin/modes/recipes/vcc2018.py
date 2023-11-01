import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes import download_vcc2018mos, prepare_vcc2018mos
from lhotse.utils import Pathlike

__all__ = ["vcc2018mos"]


@prepare.command()
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option("-nj", "--num_jobs", default=1, type=int)
def vcc2018mos(corpus_dir: Pathlike, output_dir: Pathlike, num_jobs: int):
    """VCC2018 data preparation for Mean Opinion Score (MOS) prediction."""
    prepare_vcc2018mos(corpus_dir, output_dir=output_dir, num_jobs=num_jobs)


@download.command()
@click.argument("target_dir", type=click.Path())
def vcc2018mos(target_dir: Pathlike):
    """VCC 2018 download for Mean Opinion Score (MOS) prediction."""
    download_vcc2018mos(target_dir)
