from typing import Dict, List, Optional, Tuple, Union

import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.librilight import prepare_librilight
)
from lhotse.utils import Pathlike


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
def librilight(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
):
    """LibriLight data preparation."""
    prepare_librilight(
        corpus_dir=corpus_dir,
        output_dir=output_dir,
    )
