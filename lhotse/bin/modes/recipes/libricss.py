import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.libricss import prepare_libricss
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
@click.argument("target_dir", type=click.Path())
def libricss(target_dir: Pathlike):
    """Just print the help message."""
    help(prepare_libricss)
