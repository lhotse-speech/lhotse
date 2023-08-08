import logging

import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.audio_mnist import download_audio_mnist, prepare_audio_mnist
from lhotse.utils import Pathlike


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
@click.option(
    "--force-download",
    type=bool,
    default=False,
    help="If True, download even if file is present.",
)
def audio_mnist(target_dir: Pathlike, force_download: bool):
    """AudioMNIST dataset download."""
    download_audio_mnist(target_dir, force_download)


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
def audio_mnist(corpus_dir: Pathlike, output_dir: Pathlike):
    """AudioMNIST corpus data preparation."""
    logging.basicConfig(level=logging.INFO)
    prepare_audio_mnist(
        corpus_dir,
        output_dir=output_dir,
    )
