import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.ali_meeting import download_ali_meeting, prepare_ali_meeting
from lhotse.utils import Pathlike

__all__ = ["ali_meeting"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option("--mic", type=click.Choice(["near", "far"]), default="far")
def ali_meeting(corpus_dir: Pathlike, output_dir: Pathlike, mic: str):
    """AliMeeting data preparation."""
    prepare_ali_meeting(corpus_dir, output_dir=output_dir, mic=mic)


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
@click.option("--force-download", is_flag=True, default=False)
def ali_meeting(target_dir: Pathlike, force_download: bool):
    """AliMeeting download."""
    download_ali_meeting(target_dir, force_download=force_download)
