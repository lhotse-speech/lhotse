import click

from lhotse.bin.modes import prepare
from lhotse.recipes.aspire import prepare_aspire
from lhotse.utils import Pathlike

__all__ = ["aspire"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option("--mic", type=click.Choice(["single", "multi"]), default="single")
def aspire(corpus_dir: Pathlike, output_dir: Pathlike, mic: str):
    """ASpIRE data preparation."""
    prepare_aspire(corpus_dir, output_dir=output_dir, mic=mic)
