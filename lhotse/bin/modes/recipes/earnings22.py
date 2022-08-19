import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.earnings22 import download_earnings22, prepare_earnings22
from lhotse.utils import Pathlike


@download.command(context_settings=dict(show_default=True))
def earnings22():
    """Earnings22 dataset download."""
    download_earnings22(None)


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "--normalize-text/--no-normalize-text", default=False, help="Normalize the text."
)
def earnings22(corpus_dir: Pathlike, output_dir: Pathlike, normalize_text: bool):
    """Earnings22 data preparation."""
    prepare_earnings22(corpus_dir, output_dir=output_dir, normalize_text=normalize_text)
