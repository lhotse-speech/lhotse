import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.libricss import download_libricss, prepare_libricss
from lhotse.utils import Pathlike


@prepare.command()
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "--type",
    type=click.Choice(["ihm", "ihm-mix", "mdm"]),
    default="mdm",
    help="Type of the corpus to prepare",
)
def libricss(corpus_dir: Pathlike, output_dir: Pathlike, type: str = "replay"):
    """
    LibriCSS recording and supervision manifest preparation.
    """
    prepare_libricss(corpus_dir, output_dir, type)


@download.command()
@click.argument("target_dir", type=click.Path(exists=True, dir_okay=True))
@click.option("--force-download", is_flag=True, help="Force download")
def libricss(target_dir: Pathlike, force_download: bool = False):
    """
    Download LibriCSS dataset.
    """
    download_libricss(target_dir, force_download)
