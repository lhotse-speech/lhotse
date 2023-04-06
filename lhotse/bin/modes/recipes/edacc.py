from typing import Sequence

import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.edacc import download_edacc, prepare_edacc
from lhotse.utils import Pathlike

__all__ = ["edacc"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "--alignments-dir",
    type=click.Path(exists=True, dir_okay=True),
    default=None,
    help="Path to the directory with the alignments (optional).",
)
@click.option(
    "-p",
    "--dataset-parts",
    type=str,
    default=["auto"],
    multiple=True,
    help="List of dataset parts to prepare. To prepare multiple parts, pass each with `-p` "
    "Example: `-p train-clean-360 -p dev-other`",
)
@click.option(
    "-j",
    "--num-jobs",
    type=int,
    default=1,
    help="How many threads to use (can give good speed-ups with slow disks).",
)
def edacc(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    alignments_dir: Pathlike,
    dataset_parts: Sequence[str],
    num_jobs: int,
):
    """The Edinburgh International Accents of English Corpus (EDACC) data preparation."""
    if len(dataset_parts) == 1:
        dataset_parts = dataset_parts[0]
    prepare_edacc(
        corpus_dir,
        output_dir=output_dir,
        alignments_dir=alignments_dir,
        num_jobs=num_jobs,
        dataset_parts=dataset_parts,
    )


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
def edacc(target_dir: Pathlike):
    """The Edinburgh International Accents of English Corpus (EDACC) download."""
    download_edacc(target_dir)
