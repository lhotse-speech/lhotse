from typing import Sequence

import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.librispeech import download_librispeech, prepare_librispeech
from lhotse.utils import Pathlike

__all__ = ["librispeech"]


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
@click.option(
    "--normalize-text",
    type=click.Choice(["none", "lower"], case_sensitive=False),
    default="none",
    help="Conversion of transcripts to lower-case (originally in upper-case).",
    show_default=True,
)
def librispeech(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    alignments_dir: Pathlike,
    dataset_parts: Sequence[str],
    normalize_text: str,
    num_jobs: int,
):
    """(Mini) Librispeech ASR data preparation."""
    if len(dataset_parts) == 1:
        dataset_parts = dataset_parts[0]
    prepare_librispeech(
        corpus_dir,
        output_dir=output_dir,
        alignments_dir=alignments_dir,
        num_jobs=num_jobs,
        dataset_parts=dataset_parts,
        normalize_text=normalize_text,
    )


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
@click.option(
    "--full/--mini",
    default=True,
    help="Download Librispeech [default] or mini Librispeech.",
)
def librispeech(target_dir: Pathlike, full: bool):
    """(Mini) Librispeech download."""
    download_librispeech(
        target_dir, dataset_parts="librispeech" if full else "mini_librispeech"
    )
