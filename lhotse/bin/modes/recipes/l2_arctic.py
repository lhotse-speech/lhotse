import click

from lhotse.bin.modes import prepare
from lhotse.recipes.l2_arctic import prepare_l2_arctic
from lhotse.utils import Pathlike

__all__ = ["l2_arctic"]


@prepare.command()
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
def l2_arctic(corpus_dir: Pathlike, output_dir: Pathlike):
    """L2 Arctic data preparation."""
    prepare_l2_arctic(corpus_dir, output_dir=output_dir)
