import click

from lhotse.bin.modes import prepare
from lhotse.recipes.broadcast_news import prepare_broadcast_news
from lhotse.utils import Pathlike


@prepare.command()
@click.argument("audio_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("transcript_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("output_dir", type=click.Path())
def broadcast_news(audio_dir: Pathlike, transcript_dir: Pathlike, output_dir: Pathlike):
    """
    English Broadcast News 1997 data preparation.
    It will output three manifests: for recordings, topic sections, and speech segments.
    It supports the following LDC distributions:

    \b
    * 1997 English Broadcast News Train (HUB4)
        Speech       LDC98S71
        Transcripts  LDC98T28

    This data is not available for free - your institution needs to have an LDC subscription.
    """
    prepare_broadcast_news(
        audio_dir=audio_dir, transcripts_dir=transcript_dir, output_dir=output_dir
    )
