import click

from lhotse.bin.modes import prepare
from lhotse.recipes.reazonspeech import prepare_reazonspeech
from lhotse.utils import Pathlike

__all__ = ["reazonspeech"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
def reazonspeech(corpus_dir: Pathlike, output_dir: Pathlike):
    """ReazonSpeech data preparation."""
    prepare_reazonspeech(corpus_dir, output_dir=output_dir)
