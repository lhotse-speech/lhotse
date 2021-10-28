import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.libricss import download_libricss, prepare_libricss
from lhotse.utils import Pathlike


@prepare.command()
@click.argument("corpus_zip", type=click.Path(exists=True, file_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "--type",
    type=click.Choice(["replay", "mix"]),
    default="replay",
    help="Type of the corpus to prepare",
)
def libricss(corpus_zip: Pathlike, output_dir: Pathlike, type: str = "replay"):
    """
    LibriCSS recording and supervision manifest preparation.
    """
    prepare_libricss(corpus_zip, output_dir, type)


@download.command()
@click.argument("target_dir", type=click.Path(exists=True, dir_okay=True))
def libricss(target_dir: Pathlike):
    """
    Download LibriCSS dataset.
    """
    download_libricss(target_dir)
