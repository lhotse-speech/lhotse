import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes import download_adept, prepare_adept
from lhotse.utils import Pathlike


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
def adept(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
):
    """ADEPT prosody transfer evaluation corpus data preparation."""
    prepare_adept(corpus_dir, output_dir=output_dir)


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
def adept(
    target_dir: Pathlike,
):
    """ADEPT prosody transfer evaluation corpus download."""
    download_adept(target_dir)
