import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes import download_libritts, prepare_libritts
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
def libritts(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    num_jobs: int,
    link_previous_utterance: bool,
):
    """LibriTTs data preparation."""
    prepare_libritts(
        corpus_dir,
        output_dir=output_dir,
        num_jobs=num_jobs,
        link_previous_utt=link_previous_utterance,
    )


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
def libritts(
    target_dir: Pathlike,
):
    """LibriTTS data download."""
    download_libritts(target_dir)
