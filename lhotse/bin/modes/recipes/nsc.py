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
@click.option(
    "-j",
    "--num-jobs",
    type=int,
    default=1,
    help="How many threads to use (can give good speed-ups with slow disks).",
)
def nsc(corpus_dir: Pathlike, output_dir: Pathlike, dataset_part: str, num_jobs: int):
    """
    \b
    This is a data preparation recipe for the National Corpus of Speech in Singaporean English.
    CORPUS_DIR: root directory that contains all NSC shared folder. Eg.
        ├── IMDA - National Speech Corpus
        │   ├── LEXICON
        │   ├── PART1
        │   ├── PART2
        │   └── PART3
        ├── IMDA - National Speech Corpus - Additional
        │   └── IMDA - National Speech Corpus (Additional)
        │      ├── PART4
        │      ├── PART5
        │      └── PART6
    """
    prepare_nsc(
        corpus_dir, dataset_part=dataset_part, output_dir=output_dir, num_jobs=num_jobs
    )
