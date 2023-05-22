import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.icsi import download_icsi, prepare_icsi
from lhotse.utils import Pathlike

__all__ = ["icsi"]


@download.command(context_settings=dict(show_default=True))
@click.argument("audio_dir", type=click.Path())
@click.option(
    "--transcripts-dir",
    type=click.Path(),
    default=None,
    help="To download annotations in a different directory than audio.",
)
@click.option(
    "--mic",
    type=click.Choice(["ihm", "ihm-mix", "sdm", "mdm"], case_sensitive=False),
    default="ihm",
    help="ICSI microphone setting.",
)
@click.option(
    "--url",
    type=str,
    default="http://groups.inf.ed.ac.uk/ami",
    help="ICSI data downloading URL.",
)
@click.option(
    "--force-download",
    type=bool,
    default=False,
    help="If True, download even if file is present.",
)
def icsi(
    audio_dir: Pathlike,
    transcripts_dir: Pathlike,
    mic: str,
    url: str,
    force_download: bool,
):
    """ICSI data download."""
    download_icsi(
        audio_dir,
        transcripts_dir=transcripts_dir,
        mic=mic,
        url=url,
        force_download=force_download,
    )


@prepare.command(context_settings=dict(show_default=True))
@click.argument("audio_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "--transcripts-dir", type=click.Path(exists=True, dir_okay=True), default=None
)
@click.option(
    "--mic",
    type=click.Choice(["ihm", "ihm-mix", "sdm", "mdm"], case_sensitive=False),
    default="ihm",
    help="ICSI microphone setting.",
)
@click.option(
    "--normalize-text",
    type=click.Choice(["none", "upper", "kaldi"], case_sensitive=False),
    default="kaldi",
    help="Type of text normalization to apply (kaldi style, by default)",
)
@click.option(
    "--save-to-wav",
    is_flag=True,
    default=False,
    help="If True and `mic` is sdm/ihm/mdm, save the recordings as WAV for faster processing.",
)
def icsi(
    audio_dir: Pathlike,
    transcripts_dir: Pathlike,
    output_dir: Pathlike,
    mic: str,
    normalize_text: bool,
    save_to_wav: bool,
):
    """ICSI data preparation."""
    prepare_icsi(
        audio_dir,
        transcripts_dir,
        output_dir=output_dir,
        mic=mic,
        normalize_text=normalize_text,
        save_to_wav=save_to_wav,
    )
