import click

from lhotse.bin.modes import prepare
from lhotse.recipes.nsc import NSC_PARTS, prepare_nsc
from lhotse.utils import Pathlike


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "-p",
    "--dataset-part",
    type=click.Choice(NSC_PARTS),
    default="PART3_SameCloseMic",
    help="Which part of NSC should be prepared",
)
def nsc(corpus_dir: Pathlike, output_dir: Pathlike, dataset_part: str):
    """
    This is a data preparation recipe for the National Corpus of Speech in Singaporean English.
    """
    prepare_nsc(corpus_dir, dataset_part=dataset_part, output_dir=output_dir)
