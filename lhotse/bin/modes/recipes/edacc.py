from typing import Sequence

import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.edacc import download_edacc, prepare_edacc
from lhotse.utils import Pathlike

__all__ = ["edacc"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
def edacc(corpus_dir: Pathlike, output_dir: Pathlike):
    """The Edinburgh International Accents of English Corpus (EDACC) data preparation."""
    prepare_edacc(corpus_dir, output_dir=output_dir)


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
def edacc(target_dir: Pathlike):
    """The Edinburgh International Accents of English Corpus (EDACC) download."""
    download_edacc(target_dir)
