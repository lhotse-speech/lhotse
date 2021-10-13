import click

from lhotse.bin.modes import prepare
from lhotse.recipes.babel import prepare_single_babel_language
from lhotse.utils import Pathlike


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
def babel(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
):
    """
    This is a data preparation recipe for the IARPA BABEL corpus
    (see: https://www.iarpa.gov/index.php/research-programs/babel).
    It should support all of the languages available in BABEL.
    It will prepare the data from the "conversational" part of BABEL.

    This script should be invoked separately for each language you want to prepare, e.g.:
    $ lhotse prepare babel /export/corpora5/Babel/IARPA_BABEL_BP_101 data/cantonese
    $ lhotse prepare babel /export/corpora5/Babel/BABEL_OP1_103 data/bengali
    """
    prepare_single_babel_language(corpus_dir, output_dir=output_dir)
