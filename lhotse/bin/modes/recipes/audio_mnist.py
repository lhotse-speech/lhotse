import logging

import click

from lhotse.bin.modes import prepare
from lhotse.recipes.audio_mnist import prepare_audio_mnist
from lhotse.utils import Pathlike


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
def audio_mnist(corpus_dir: Pathlike, output_dir: Pathlike):
    """AudioMNIST speech translation data preparation."""
    logging.basicConfig(level=logging.INFO)
    prepare_audio_mnist(
        corpus_dir,
        output_dir=output_dir,
    )
