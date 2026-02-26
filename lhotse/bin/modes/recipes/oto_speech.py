import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.oto_speech import download_oto_speech, prepare_oto_speech
from lhotse.utils import Pathlike

__all__ = ["oto_speech"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "-p",
    "--dataset-parts",
    type=str,
    multiple=True,
    default=("train",),
    help="Dataset parts to prepare.",
)
@click.option(
    "--target-sr",
    type=int,
    default=16000,
    help="Target sampling rate for lazy resampling.",
)
def oto_speech(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    dataset_parts: tuple,
    target_sr: int,
):
    """otoSpeech data preparation."""
    prepare_oto_speech(
        corpus_dir=corpus_dir,
        output_dir=output_dir,
        parts=dataset_parts,
        target_sr=target_sr,
    )


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
@click.option(
    "--force-download",
    is_flag=True,
    default=False,
    help="Force download of audio and pseudo-labels.",
)
@click.option(
    "-p",
    "--dataset-parts",
    type=str,
    multiple=True,
    default=("train",),
    help="Dataset parts to download (otoSpeech standard release only provides 'train').",
)
@click.option(
    "--version",
    type=str,
    default="full-duplex-processed-141h",
    help="Dataset version suffix on HuggingFace.",
)
def oto_speech(
    target_dir: Pathlike,
    force_download: bool,
    dataset_parts: tuple,
    version: str,
):
    """otoSpeech dataset download."""
    download_oto_speech(
        target_dir=target_dir,
        parts=dataset_parts,
        version=version,
        force_download=force_download,
    )
