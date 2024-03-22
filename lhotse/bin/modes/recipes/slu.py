from typing import List, Optional, Sequence, Tuple, Union

import click

from lhotse.bin.modes import prepare
from lhotse.recipes.slu import prepare_slu
from lhotse.utils import Pathlike


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path())
@click.argument("output_dir", type=click.Path())
def slu(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
):
    prepare_slu(corpus_dir=corpus_dir, output_dir=output_dir)
