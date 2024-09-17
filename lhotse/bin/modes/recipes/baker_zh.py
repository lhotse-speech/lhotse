import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.baker_zh import download_baker_zh, prepare_baker_zh
from lhotse.utils import Pathlike

__all__ = ["baker_zh"]


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path(), default=".")
def baker_zh(target_dir: Pathlike):
    """bazker_zh download."""
    download_baker_zh(target_dir)


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
def baker_zh(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
):
    """bazker_zh data preparation."""
    prepare_baker_zh(corpus_dir, output_dir=output_dir)
