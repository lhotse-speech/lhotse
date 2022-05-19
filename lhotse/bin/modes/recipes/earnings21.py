import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.earnings21 import download_earnings21, prepare_earnings21
from lhotse.utils import Pathlike


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
def earnings21(target_dir: Pathlike):
    """Earnings21 dataset download."""
    download_earnings21(target_dir)


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "--normalize-text/--no-normalize-text", default=False, help="Normalize the text."
)
def earnings21(corpus_dir: Pathlike, output_dir: Pathlike, normalize_text: bool):
    """Earnings21 data preparation."""
    prepare_earnings21(corpus_dir, output_dir=output_dir, normalize_text=normalize_text)
