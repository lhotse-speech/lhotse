import click

from lhotse.bin.modes import prepare
from lhotse.recipes.tal_asr import prepare_tal_asr
from lhotse.utils import Pathlike

__all__ = ["tal_asr"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
def tal_asr(corpus_dir: Pathlike, output_dir: Pathlike):
    """Tal_asr ASR data preparation."""
    prepare_tal_asr(corpus_dir, output_dir=output_dir)
