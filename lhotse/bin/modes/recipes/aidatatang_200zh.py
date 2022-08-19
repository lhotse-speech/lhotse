import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.aidatatang_200zh import (
    download_aidatatang_200zh,
    prepare_aidatatang_200zh,
)
from lhotse.utils import Pathlike

__all__ = ["aidatatang_200zh"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
def aidatatang_200zh(corpus_dir: Pathlike, output_dir: Pathlike):
    """aidatatang_200zh ASR data preparation.
    Args:
      corpus_dir:
        It should contain a subdirectory "aidatatang_200zh"
      output_dir:
        The output directory.
    """
    prepare_aidatatang_200zh(corpus_dir, output_dir=output_dir)


@download.command(context_settings=dict(show_default=True))
@click.argument(
    "target_dir",
    type=click.Path(),
)
def aidatatang_200zh(target_dir: Pathlike):
    """aidatatang_200zh download.
    Args:
      target_dir:
        It will create a dir aidatatang_200zh to contain all
        downloaded/extracted files
    """
    download_aidatatang_200zh(target_dir)
