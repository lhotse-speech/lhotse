from typing import Sequence

import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes import download_libritts, download_librittsr, prepare_libritts
from lhotse.utils import Pathlike

__all__ = ["libritts"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "-j",
    "--num-jobs",
    type=int,
    default=1,
    help="How many jobs to use (can give good speed-ups with slow disks).",
)
@click.option(
    "--link-previous-utterance/--no-previous-utterance",
    default=False,
    help=(
        "If true adds previous utterance id to supervisions. "
        "Useful for reconstructing chains of utterances as they were read from LibriVox books. "
        "If previous utterance was skipped from LibriTTS datasets previous_utt label is None. "
        "66% of utterances have previous utterance."
    ),
)
@click.option(
    "-p",
    "--dataset-parts",
    type=str,
    default=["all"],
    multiple=True,
    help="List of dataset parts to prepare. To prepare multiple parts, pass each with `-p` "
    "Example: `-p train-clean-360 -p dev-other`",
)
def libritts(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    dataset_parts: Sequence[str],
    num_jobs: int,
    link_previous_utterance: bool,
):
    """LibriTTS data preparation."""
    prepare_libritts(
        corpus_dir,
        output_dir=output_dir,
        num_jobs=num_jobs,
        dataset_parts=dataset_parts,
        link_previous_utt=link_previous_utterance,
    )


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
@click.option(
    "-p",
    "--dataset-parts",
    type=str,
    default=["all"],
    multiple=True,
    help="List of dataset parts to download. To prepare multiple parts, pass each with `-p` "
    "Example: `-p train-clean-360 -p dev-other`",
)
def libritts(
    target_dir: Pathlike,
    dataset_parts: Sequence[str],
):
    """LibriTTS data download."""
    download_libritts(target_dir, dataset_parts=dataset_parts)


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
@click.option(
    "-p",
    "--dataset-parts",
    type=str,
    default=["all"],
    multiple=True,
    help="List of dataset parts to download. To prepare multiple parts, pass each with `-p` "
    "Example: `-p train-clean-360 -p dev-other`",
)
def librittsr(
    target_dir: Pathlike,
    dataset_parts: Sequence[str],
):
    """LibriTTS-R data download."""
    download_librittsr(target_dir, dataset_parts=dataset_parts)


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "-j",
    "--num-jobs",
    type=int,
    default=1,
    help="How many jobs to use (can give good speed-ups with slow disks).",
)
@click.option(
    "--link-previous-utterance/--no-previous-utterance",
    default=False,
    help=(
        "If true adds previous utterance id to supervisions. "
        "Useful for reconstructing chains of utterances as they were read from LibriVox books. "
        "If previous utterance was skipped from LibriTTS datasets previous_utt label is None. "
        "66% of utterances have previous utterance."
    ),
)
@click.option(
    "-p",
    "--dataset-parts",
    type=str,
    default=["all"],
    multiple=True,
    help="List of dataset parts to prepare. To prepare multiple parts, pass each with `-p` "
    "Example: `-p train-clean-360 -p dev-other`",
)
def librittsr(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    dataset_parts: Sequence[str],
    num_jobs: int,
    link_previous_utterance: bool,
):
    """LibriTTS-R data preparation."""
    prepare_libritts(
        corpus_dir,
        output_dir=output_dir,
        num_jobs=num_jobs,
        dataset_parts=dataset_parts,
        link_previous_utt=link_previous_utterance,
    )
