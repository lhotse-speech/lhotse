import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.spgispeech import download_spgispeech, prepare_spgispeech
from lhotse.utils import Pathlike

__all__ = ["spgispeech"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "-j",
    "--num-jobs",
    type=int,
    default=1,
    help="How many threads to use (can give good speed-ups with slow disks).",
)
@click.option(
    "--normalize-text/--no-normalize-text", default=True, help="Normalize the text."
)
def spgispeech(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    num_jobs: int,
    normalize_text: bool,
):
    """SPGISpeech ASR data preparation."""
    prepare_spgispeech(
        corpus_dir,
        output_dir,
        num_jobs=num_jobs,
        normalize_text=normalize_text,
    )


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
def spgispeech(target_dir: Pathlike):
    """SPGISpeech download."""
    download_spgispeech(target_dir)
