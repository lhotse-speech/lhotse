import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.ali_meeting import download_ali_meeting, prepare_ali_meeting
from lhotse.utils import Pathlike

__all__ = ["ali_meeting"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "--mic", type=click.Choice(["near", "far", "ihm", "sdm", "mdm"]), default="far"
)
@click.option(
    "--normalize-text",
    type=click.Choice(["none", "m2met"], case_sensitive=False),
    default="none",
    help="Type of text normalization to apply (M2MeT style is from the official challenge)",
)
@click.option(
    "--save-mono",
    is_flag=True,
    default=False,
    help="If True and `mic` is sdm, extract first channel and save as new recording.",
)
def ali_meeting(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    mic: str,
    normalize_text: str,
    save_mono: bool,
):
    """AliMeeting data preparation."""
    prepare_ali_meeting(
        corpus_dir,
        output_dir=output_dir,
        mic=mic,
        normalize_text=normalize_text,
        save_mono=save_mono,
    )


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
@click.option("--force-download", is_flag=True, default=False)
def ali_meeting(target_dir: Pathlike, force_download: bool):
    """AliMeeting download."""
    download_ali_meeting(target_dir, force_download=force_download)
