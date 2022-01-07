import click

from lhotse.bin.modes import prepare
from lhotse.recipes.icsi import prepare_icsi
from lhotse.utils import Pathlike

__all__ = ["icsi"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("audio_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("transcript_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "--mic",
    type=click.Choice(["ihm", "sdm", "mdm"], case_sensitive=False),
    default="ihm",
    help="AMI microphone setting.",
)
@click.option(
    "--normalize-text",
    is_flag=True,
    help="If set, convert all text annotations to upper case (similar to Kaldi)",
)
def icsi(
    audio_dir: Pathlike,
    transcript_dir: Pathlike,
    output_dir: Pathlike,
    mic: str,
    normalize_text: bool,
):
    """AMI data preparation."""
    prepare_icsi(
        audio_dir,
        transcript_dir,
        output_dir=output_dir,
        mic=mic,
        normalize_text=normalize_text,
    )
