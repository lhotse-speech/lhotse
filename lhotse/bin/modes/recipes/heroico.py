import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.heroico import download_heroico, prepare_heroico
from lhotse.utils import Pathlike

__all__ = ["heroico"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("speech_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("transcript_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
def heroico(speech_dir: Pathlike, transcript_dir: Pathlike, output_dir: Pathlike):
    """heroico Answers ASR data preparation."""
    prepare_heroico(speech_dir, transcript_dir, output_dir)


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
def heroico(target_dir: Pathlike):
    """heroico download."""
    download_heroico(target_dir)
