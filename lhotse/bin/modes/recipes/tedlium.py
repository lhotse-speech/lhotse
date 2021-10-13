import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.tedlium import download_tedlium, prepare_tedlium
from lhotse.utils import Pathlike


@prepare.command()
@click.argument(
    "tedlium_dir", type=click.Path(exists=True, dir_okay=True, file_okay=False)
)
@click.argument("output_dir", type=click.Path())
def tedlium(tedlium_dir: Pathlike, output_dir: Pathlike):
    """
    TED-LIUM v3 recording and supervision manifest preparation.
    """
    prepare_tedlium(tedlium_root=tedlium_dir, output_dir=output_dir)


@download.command()
@click.argument("target_dir", type=click.Path())
def tedlium(target_dir: Pathlike):
    """TED-LIUM v3 download (approx. 11GB)."""
    download_tedlium(target_dir)
